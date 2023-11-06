import json
import os
import logging
from collections import Counter
from parser_transitions import minibatch_parse

from tqdm import tqdm
import torch
import numpy as np

P_PREFIX = '<p>:'
L_PREFIX = '<l>:'
UNK = '<UNK>'
NULL = '<NULL>'
ROOT = '<ROOT>'
PUNCTS = ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]


class Config(object):
    P_PREFIX = '<p>:'
    L_PREFIX = '<l>:'
    UNK = '<UNK>'
    NULL = '<NULL>'
    ROOT = '<ROOT>'
    with_punct = True
    unlabeled = False
    lowercase = True
    use_pos = True
    use_dep = True
    use_dep = use_dep and (not unlabeled)
    data_path = './data'
    train_file = 'train.conll'
    dev_file = 'dev.conll'
    test_file = 'test.conll'
    real_test_file = 'real_test.conll'
    vocab = json.load(open(os.path.join(data_path, 'vocab.json'), 'r',encoding='utf-8'))
    vocab_size = len(vocab)
    vector_file = data_path + '/word2vec.txt'
    hidden_size = 1000
    embedding_size = 100
    dropout = 0
    model = data_path + '/model.pt'


class Parser(object):
    """Contains everything needed for transition-based dependency parsing except for the model"""

    def __init__(self, dataset):
        root_labels = list([l for ex in dataset
                           for (h, l) in zip(ex['head'], ex['label']) if h == 0])
        counter = Counter(root_labels)
        if len(counter) > 1:
            logging.info('Warning: more than one root label')
            logging.info(counter)
        self.root_label = counter.most_common()[0][0]
        deprel = [self.root_label] + list(set([w for ex in dataset
                                               for w in ex['label']
                                               if w != self.root_label]))
        tok2id = {L_PREFIX + l: i for (i, l) in enumerate(deprel)}
        tok2id[L_PREFIX + NULL] = self.L_NULL = len(tok2id)

        config = Config()
        self.config = config
        self.unlabeled = config.unlabeled
        self.with_punct = config.with_punct
        self.use_pos = config.use_pos
        self.use_dep = config.use_dep

        if self.unlabeled:
            trans = ['L', 'R', 'S', 'D']
            self.n_deprel = 1
        else:
            trans = ['L-' + l for l in deprel] + ['R-' + l for l in deprel] + ['S', 'D']
            self.n_deprel = len(deprel)

        self.deprel = deprel
        self.id2deprel = {i: w for (i, w) in enumerate(deprel)}
        self.n_trans = len(trans)
        self.tran2id = {t: i for (i, t) in enumerate(trans)}
        self.id2tran = {i: t for (i, t) in enumerate(trans)}

        # logging.info('Build dictionary for part-of-speech tags.')
        tok2id.update(build_dict([P_PREFIX + w for ex in dataset for w in ex['pos']],
                                 offset=len(tok2id)))
        tok2id[P_PREFIX + UNK] = self.P_UNK = len(tok2id)
        tok2id[P_PREFIX + NULL] = self.P_NULL = len(tok2id)
        tok2id[P_PREFIX + ROOT] = self.P_ROOT = len(tok2id)

        # logging.info('Build dictionary for words.')
        tok2id.update(build_dict([w for ex in dataset for w in ex['word']],
                                 offset=len(tok2id)))
        tok2id[UNK] = self.UNK = len(tok2id)
        tok2id[NULL] = self.NULL = len(tok2id)
        tok2id[ROOT] = self.ROOT = len(tok2id)

        self.tok2id = tok2id
        self.id2tok = {v: k for (k, v) in tok2id.items()}

        self.n_tokens = len(tok2id)
        self.model = None

    def vectorize(self, examples):
        vec_examples = []
        for ex in examples:
            word = [self.ROOT] + [self.tok2id[w] if w in self.tok2id
                                  else self.UNK for w in ex['word']]
            pos = [self.P_ROOT] + [self.tok2id[P_PREFIX + w] if P_PREFIX + w in self.tok2id
                                   else self.P_UNK for w in ex['pos']]
            head = [-1] + ex['head']
            label = [-1] + [self.tok2id[L_PREFIX + w] if L_PREFIX + w in self.tok2id
                            else -1 for w in ex['label']]
            vec_examples.append({'word': word, 'pos': pos,
                                 'head': head, 'label': label})
        return vec_examples

    def extract_features(self, stack, buf, arcs, ex):
        # TODO:
        # You should implement your feature extraction here.
        # Extract the features for one example, ex
        # The features could include the word itself, the part-of-speech and so on.
        # Every feature could be represented by a string,
        # and the string can be converted to an id(int), according to the self.tok2id
        # Return: A list of token_ids corresponding to tok2id

        # Cite: A Fast and Accurate Dependency Parser using Neural Networks by Danqi Chen and Christopher Manning
        # arc: (head, tail, label)
        # feature: The top 3 words on the stack and buffer; The first and second leftmost / rightmost
        # children of the top two words on the stack; The leftmost of leftmost / rightmost of rightmost children of the top two words on the stack:
        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k], reverse=True)

        if stack[0] == "ROOT":
            stack[0] = 0

        features = [self.NULL] * (3-len(stack)) + [ex['word'][x] for x in stack[-3:]] # 3
        features += [ex['word'][x] for x in buf[:3]] + [self.NULL] * (3-len(buf)) # 3
        pos_features = []
        label_features = []

        if self.use_pos:
            # 6
            pos_features = [self.P_NULL] * (3-len(stack)) + [ex['pos'][x] for x in stack[-3:]]
            pos_features += [ex['pos'][x] for x in buf[:3]] + [self.P_NULL] * (3-len(buf))

        for i in range(2):
            # 2 * 36 = 72
            # 2 * 24 = 48
            if i < len(stack):  
                si = stack[-i-1]
                lc = get_lc(si)
                rc = get_rc(si)
                llc = get_lc(lc[0]) if len(lc) > 0 else []
                rrc = get_rc(rc[0]) if len(rc) > 0 else []

                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.NULL)
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.NULL)
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.NULL)
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.NULL)
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.NULL)
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.NULL)

                if self.use_pos:
                    pos_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.P_NULL)
                    pos_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.P_NULL)
                    pos_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.P_NULL)
                    pos_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.P_NULL)
                    pos_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.P_NULL)
                    pos_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.P_NULL)

                if self.use_dep:
                    label_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.L_NULL)
                    label_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.L_NULL)
                    label_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.L_NULL)
                    label_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.L_NULL)
                    label_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.L_NULL)
                    label_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.L_NULL)
            else:
                features += [self.NULL] * 6
                if self.use_pos:
                    pos_features += [self.P_NULL] * 6
                if self.use_dep:
                    label_features += [self.L_NULL] * 6

            if i < len(buf):
                si = buf[i]
                lc = get_lc(si)
                rc = get_rc(si)
                llc = get_lc(lc[0]) if len(lc) > 0 else []
                rrc = get_rc(rc[0]) if len(rc) > 0 else []

                features.append(ex['word'][lc[0]] if len(lc) > 0 else self.NULL)
                features.append(ex['word'][rc[0]] if len(rc) > 0 else self.NULL)
                features.append(ex['word'][lc[1]] if len(lc) > 1 else self.NULL)
                features.append(ex['word'][rc[1]] if len(rc) > 1 else self.NULL)
                features.append(ex['word'][llc[0]] if len(llc) > 0 else self.NULL)
                features.append(ex['word'][rrc[0]] if len(rrc) > 0 else self.NULL)

                if self.use_pos:
                    pos_features.append(ex['pos'][lc[0]] if len(lc) > 0 else self.P_NULL)
                    pos_features.append(ex['pos'][rc[0]] if len(rc) > 0 else self.P_NULL)
                    pos_features.append(ex['pos'][lc[1]] if len(lc) > 1 else self.P_NULL)
                    pos_features.append(ex['pos'][rc[1]] if len(rc) > 1 else self.P_NULL)
                    pos_features.append(ex['pos'][llc[0]] if len(llc) > 0 else self.P_NULL)
                    pos_features.append(ex['pos'][rrc[0]] if len(rrc) > 0 else self.P_NULL)

                if self.use_dep:
                    label_features.append(ex['label'][lc[0]] if len(lc) > 0 else self.L_NULL)
                    label_features.append(ex['label'][rc[0]] if len(rc) > 0 else self.L_NULL)
                    label_features.append(ex['label'][lc[1]] if len(lc) > 1 else self.L_NULL)
                    label_features.append(ex['label'][rc[1]] if len(rc) > 1 else self.L_NULL)
                    label_features.append(ex['label'][llc[0]] if len(llc) > 0 else self.L_NULL)
                    label_features.append(ex['label'][rrc[0]] if len(rrc) > 0 else self.L_NULL)        
            else:
                features += [self.NULL] * 6
                if self.use_pos:
                    pos_features += [self.P_NULL] * 6
                if self.use_dep:
                    label_features += [self.L_NULL] * 6

        features += pos_features + label_features
        return features

    def get_oracle(self, stack, buf, ex, dependency):
        # print(stack, buf, dependency)
        # TODO: 根据当前状态，返回应该执行的操作编号（对应__init__中的trans），若无操作则返回None。
        if len(stack) < 1:
            return (self.n_trans - 2, (-1, -1)) if len(buf) > 0 else (None, (-1, -1))
        if len(buf) < 1:
            e1 = stack[-1]
            if (e1 > 0) and (dependency[e1] != -1) and (all(dependency[dep] == e1 for dep in range(len(ex['head'])) if ex['head'][dep] == e1)):
                return (self.n_trans - 1, (-1, -1))
            else:
                return (None, (-1, -1))
            return (self.n_trans - 1, (-1, -1)) if len(stack) >= 1 else (None, (-1, -1))
        # if len(buf) == 0:
        #     return self.n_trans - 1 if len(stack) > 0 else None
        # e0 = stack[-1]
        e0 = buf[0]
        e1 = stack[-1]
        h0 = ex['head'][e0]
        h1 = ex['head'][e1]
        l0 = ex['label'][e0]
        l1 = ex['label'][e1]
        
        if self.unlabeled:
            if (e1 > 0) and (h1 == e0):
                return (0, (e1, e0))
            elif (e1 >= 0) and (h0 == e1):
                return (1, (e0, e1))
            elif (e1 > 0) and (dependency[e1] != -1) and (all(dependency[dep] == e1 for dep in range(len(ex['head'])) if ex['head'][dep] == e1)):
                return (3, (-1, -1))
            else:
                return (None, (-1, -1)) if len(buf) == 0 else (2, (-1, -1))
        else:
            if (e1 > 0) and (h1 == e0):
                return (l1, (e1, e0)) if (l1 >= 0) and (l1 < self.n_deprel) else (None, (-1, -1))
            elif (e1 >= 0) and (h0 == e1):
                return (l0 + self.n_deprel, (e0, e1)) if (l0 >= 0) and (l0 < self.n_deprel) else (None, (-1, -1))
            elif (e1 > 0) and (dependency[e1] != -1) and (all(dependency[dep] == e1 for dep in range(len(ex['head'])) if ex['head'][dep] == e1)):
                return (self.n_trans - 1, (-1, -1))
            else:
                return (None, (-1, -1)) if len(buf) == 0 else (self.n_trans - 2, (-1, -1))
        # if self.unlabeled:
        #     if (e1 > 0) and (h1 == e0):
        #         return 0
        #     elif (e1 >= 0) and (h0 == e1) and \
        #          (not any([x for x in buf if ex['head'][x] == e0])):
        #         return 1
        #     else:
        #         return None if len(buf) == 0 else 2
        # else:
        #     if (e1 > 0) and (h1 == e0):
        #         return l1 if (l1 >= 0) and (l1 < self.n_deprel) else None
        #     elif (e1 >= 0) and (h0 == e1) and \
        #          (not any([x for x in buf if ex['head'][x] == e0])):
        #         return l0 + self.n_deprel if (l0 >= 0) and (l0 < self.n_deprel) else None
        #     else:
        #         return None if len(buf) == 0 else self.n_trans - 1

    def create_instances(self, examples):
        all_instances = []
        succ = 0
        for id, ex in enumerate(examples):
            n_words = len(ex['word']) - 1

            # arcs = {(h, t, label)}
            stack = [0]
            buf = [i + 1 for i in range(n_words)]
            arcs = []
            instances = []
            dependency = [-1 for i in range(n_words + 1)]
            # print(ex)
            for i in range(n_words * 2):
                # print(stack, buf, arcs,dependency)
                gold_t, dep = self.get_oracle(stack, buf, ex, dependency)
                # print(gold_t, dependency[stack[-1]])
                if dep != (-1, -1):
                    dependency[dep[0]] = dep[1]
                if gold_t is None:
                    break
                legal_labels = self.legal_labels(stack, buf, arcs)
                assert legal_labels[gold_t] == 1
                instances.append((self.extract_features(stack, buf, arcs, ex),
                                  legal_labels, gold_t))
                # TODO: 根据gold_t，更新stack, arcs, buf
                # if gold_t == self.n_trans - 1:
                #     stack.append(buf[0])
                #     buf = buf[1:]
                # elif gold_t < self.n_deprel:
                #     arcs.append((stack[-1], stack[-2], gold_t))
                #     stack = stack[:-2] + [stack[-1]]
                # else:
                #     arcs.append((stack[-2], stack[-1], gold_t - self.n_deprel))
                #     stack = stack[:-1]
                if gold_t == self.n_trans - 2:
                    stack.append(buf.pop(0))
                elif gold_t == self.n_trans - 1:
                    stack.pop()
                elif gold_t < self.n_deprel:
                    arcs.append((buf[0], stack[-1], gold_t))
                    stack.pop()
                else:
                    arcs.append((stack[-1], buf[0], gold_t - self.n_deprel))
                    stack.append(buf.pop(0))
            else:
                succ += 1
                all_instances += instances
            
        return all_instances

    def legal_labels(self, stack, buf, arcs):
        # labels = ([1] if len(stack) > 2 else [0]) * self.n_deprel
        # labels += ([1] if len(stack) >= 2 else [0]) * self.n_deprel
        # labels += [1] if len(buf) > 0 else [0]
        labels = ([1] if (len(stack) > 1) and (len(buf) >= 1) else [0]) * self.n_deprel
        labels += ([1] if (len(stack) >= 1) and (len(buf) >= 1) else [0]) * self.n_deprel
        labels += [1] if len(buf) > 0 else [0]
        can_reduce = False
        for arc in arcs:
            if (len(stack) > 1) and (arc[1] == stack[-1]):
                can_reduce = True
                break
        labels += [1] if (len(stack) > 1) and can_reduce else [0]
        return labels

    def parse(self, dataset, eval_batch_size=5000):
        sentences = []
        sentence_id_to_idx = {}
        for i, example in enumerate(dataset):
            n_words = len(example['word']) - 1
            sentence = [j + 1 for j in range(n_words)]
            sentences.append(sentence)
            sentence_id_to_idx[id(sentence)] = i

        model = ModelWrapper(self, dataset, sentence_id_to_idx)
        dependencies = minibatch_parse(sentences, model, eval_batch_size)

        UAS = all_tokens = 0.0
        with tqdm(total=len(dataset)) as prog:
            for i, ex in enumerate(dataset):
                head = [-1] * len(ex['word'])
                for h, t, l in dependencies[i]:
                    head[t] = h
                for pred_h, gold_h, gold_l, pos in \
                        zip(head[1:], ex['head'][1:], ex['label'][1:], ex['pos'][1:]):
                    assert self.id2tok[pos].startswith(P_PREFIX)
                    pos_str = self.id2tok[pos][len(P_PREFIX):]
                    if (self.with_punct) or (not (pos_str in PUNCTS)):
                        UAS += 1 if pred_h == gold_h else 0
                        all_tokens += 1
                prog.update(i + 1)
        UAS /= all_tokens
        return UAS, dependencies


class ModelWrapper(object):
    def __init__(self, parser, dataset, sentence_id_to_idx):
        self.parser = parser
        self.dataset = dataset
        self.sentence_id_to_idx = sentence_id_to_idx
    def predict(self, partial_parses):
        mb_x = [self.parser.extract_features(p.stack, p.buffer, p.dependencies,
                                             self.dataset[self.sentence_id_to_idx[id(p.sentence)]])
                for p in partial_parses]
        mb_x = np.array(mb_x).astype('int32')
        mb_x = torch.from_numpy(mb_x).long()
        mb_l = [self.parser.legal_labels(p.stack, p.buffer, p.dependencies) for p in partial_parses]
        mb_w = [([10000] * self.parser.n_deprel) + ([10000] * self.parser.n_deprel) + [10000, 10000] for p in partial_parses ]
        pred = self.parser.model(mb_x)
        pred = pred.cpu().detach().numpy()
        #print(mb_l[0],mb_w[0], np.multiply(np.array(mb_l), np.array(mb_w))[0])
        # pred = np.argsort(pred)
        # result = [None for _ in len(preds)]
        # for pred, p in zip(pred, partial_parses):
        #     for item in pred:
        #         if item <  and len(p.stack) > 0 and len(p.buf) > 0:

        pred = np.argmax(pred + 10000*np.array(mb_l).astype('float32'), 1)
        #print(pred[0])
        # label = np.argmax(label, 1)
        pred = [self.parser.id2tran[p] for p in pred]
        # label = self.parser.id2deprel[label]
        return pred

def read_conll(in_file, lowercase=False, max_example=None):
    examples = []
    with open(in_file) as f:
        word, pos, head, label = [], [], [], []
        for line in f.readlines():
            sp = line.strip().split('\t')
            if len(sp) == 10:
                if '-' not in sp[0]:
                    word.append(sp[1].lower() if lowercase else sp[1])
                    pos.append(sp[4])
                    head.append(int(sp[6]))
                    label.append(sp[7])
            elif len(word) > 0:
                examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
                word, pos, head, label = [], [], [], []
                if (max_example is not None) and (len(examples) == max_example):
                    break
        if len(word) > 0:
            examples.append({'word': word, 'pos': pos, 'head': head, 'label': label})
    return examples


def build_dict(keys, n_max=None, offset=0):
    count = Counter()
    for key in keys:
        count[key] += 1
    ls = count.most_common() if n_max is None \
        else count.most_common(n_max)

    return {w[0]: index + offset for (index, w) in enumerate(ls)}


def load_and_preprocess_data(reduced=False):
    config = Config()

    print("Loading data...",)
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    dev_set = read_conll(os.path.join(config.data_path, config.dev_file),
                         lowercase=config.lowercase)
    test_set = read_conll(os.path.join(config.data_path, config.test_file),
                          lowercase=config.lowercase)
    if reduced:
        train_set = train_set[:1000]
        dev_set = dev_set[:500]
        test_set = test_set[:500]

    print("Building parser...",)
    parser = Parser(train_set)

    print("Loading Embeddings...",)
    word_vectors = {}
    for line in open(config.vector_file).readlines():
        sp = line.strip().split()
        word_vectors[sp[0]] = [float(x) for x in sp[1:]]
    embeddings_matrix = np.asarray(np.random.normal(0, 0.9, (parser.n_tokens, 50)), dtype='float32')

    for token in parser.tok2id:
        i = parser.tok2id[token]
        if token in word_vectors:
            embeddings_matrix[i] = word_vectors[token]
        elif token.lower() in word_vectors:
            embeddings_matrix[i] = word_vectors[token.lower()]

    print("Vectorizing data...",)
    train_set = parser.vectorize(train_set)
    dev_set = parser.vectorize(dev_set)
    test_set = parser.vectorize(test_set)

    print("Preprocessing training data...",)
    train_examples = parser.create_instances(train_set)
    return parser, embeddings_matrix, train_examples, dev_set, test_set,
