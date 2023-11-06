import json

def get_data(raw_data_file, new_data_file):
    sentence = []
    with open(raw_data_file, 'r', encoding='utf-8') as f:
        word, pos, head, label = [], [], [], []
        for line in f:
            item = line.strip().split()
            if len(item) == 10:
                word.append(item[1])
                pos.append(item[4])
                head.append(int(item[6]))
                label.append(item[7])
            else:
                sentence.append({
                    'word': word,
                    'pos': pos,
                    'head': head,
                    'label': label
                })
                word, pos, head, label = [], [], [], []
    f.close()

    with open(new_data_file, 'a', encoding='utf-8') as fs:
        for s in sentence:
            json.dump(s, fs, ensure_ascii=False)
            fs.writelines('\n')
    fs.close()
    print(len(sentence))

def get_vocab(data_file, vocab_file):
    words = [
        "<UNK>", "<ROOT>", "<NULL>", "<P><UNK>", "<P><NULL>", "<P><ROOT>",
        "<l><NULL>"
    ]
    words_num, pos_num, label_num = 0, 0, 0
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            word = data["word"]
            pos = data["pos"]
            label = data["label"]

            for w in word:
                if w not in words:
                    words.append(w)
                    words_num += 1

            for p in pos:
                if "<p>" + p not in words:
                    words.append("<p>" + p)
                    pos_num += 1

            for l in label:
                if "<l>" + l not in words:
                    words.append("<l>" + l)
                    label_num += 1

    f.close()
    words2id = {j: i for i, j in enumerate(words)}
    with open(vocab_file, 'w', encoding='utf-8') as fs:
        fs.write(json.dumps(words2id, ensure_ascii=False, indent=4))
    fs.close()

    print("words size [%d]" % words_num)
    print("pos size [%d]" % pos_num)
    print("label size [%d]" % label_num)
    '''
    words size [34571]
    pos size [31]
    label size [11]
    '''

if __name__=="__main__":
    raw_data_file = './data/train.conll'
    new_data_file = "./data/new_train.json"
    get_data(raw_data_file,new_data_file)
    vocab = './data/vocab.json'
    get_vocab(new_data_file,vocab)