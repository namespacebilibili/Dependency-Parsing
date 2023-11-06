from datetime import datetime
import os

from torch import optim
import torch

from trainer import ParserTrainer
from parsing_model import ParsingModel
from parser_utils import read_conll, Config, Parser
from utils import evaluate
import json
import numpy as np

def load_and_preprocess_real_data():
    config = Config()

    print("Loading data...",)
    train_set = read_conll(os.path.join(config.data_path, config.train_file),
                           lowercase=config.lowercase)
    real_test_set = read_conll(os.path.join(config.data_path, config.real_test_file),
                          lowercase=config.lowercase)

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
    real_test_set = parser.vectorize(real_test_set)

    return parser, embeddings_matrix, real_test_set
if __name__ == "__main__":
    config = Config()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, real_test_data = load_and_preprocess_real_data()
    parser.model = ParsingModel(embeddings, parser.n_trans, config).to(device) # You can add more arguments, depending on how you designed your parsing model
    input_dir = "results/2layer_1000_1/"
    input_path = input_dir + "model.weights"


    print(80 * "=")
    print("TESTING")
    print(80 * "=")
    parser.model.load_state_dict(torch.load(input_path))
    parser.model.to(device)
    print("Evaluation on real test set", )
    parser.model.eval()

    UAS, dependencies = parser.parse(real_test_data)
    with open('./prediction.json', 'w') as fh:
        json.dump(dependencies, fh)
    temp = [[] for _ in range(len(dependencies))]
    idx = 0
    for dependency in dependencies:
        temp[idx] = [(0, 0) for _ in range(len(dependency))]
        for (x, y, z) in dependency:
            temp[idx][y - 1] = (x - 1, parser.tok2id[parser.config.L_PREFIX + z])
        idx += 1
    dependencies = temp
    gold_dependencies = [[] for _ in range(len(real_test_data))]
    for i, ex in enumerate(real_test_data):
        gold_dependencies[i] = [(0, 0) for _ in range(len(ex["word"][1:]))]
        cnt = 0
        for h, l in zip(ex["head"][1:], ex["label"][1:]):
            gold_dependencies[i][cnt]=(h - 1, l)
            cnt += 1
    uas,las = evaluate(dependencies, gold_dependencies)  # To check the format of the input, please refer to the utils.py
    print("- test UAS: {:.2f}".format(uas * 100.0), "- test las: {:.2f}".format(las * 100.0))
    print("Done!")
