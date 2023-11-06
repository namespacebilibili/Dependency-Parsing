from datetime import datetime
import os

from torch import optim
import torch
import torch.nn as nn
from trainer import ParserTrainer
from parsing_model import ParsingModel
from parser_utils import load_and_preprocess_data, Config
import json
from utils import evaluate

if __name__ == "__main__":
    # Note: Set debug to False, when training on entire corpus
    # debug = True
    debug = False
    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)
    config = Config()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.model = ParsingModel(embeddings, parser.n_trans, config).to(device) # You can add more arguments, depending on how you designed your parsing model

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # TODO:
    # 1. Call an optimizer (no need to specify parameters yet, which will be implemented during training)
    # 2. Construct the Cross Entropy Loss Function in variable `loss_func`

    optimizer = optim.Adam(parser.model.parameters(), lr=0.0005, weight_decay = 1e-7)
    loss_func = nn.CrossEntropyLoss()
    trainer = ParserTrainer(
        train_data=train_data,
        dev_data=dev_data,
        optimizer=optimizer,
        loss_func=loss_func,
        output_path=output_path,
        batch_size=1024,
        n_epochs=15,
        lr=0.0005,
        weight_decay = 1e-8
    )
    trainer.train(parser, )

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path))
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data)
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
        gold_dependencies = [[] for _ in range(len(test_data))]
        for i, ex in enumerate(test_data):
            gold_dependencies[i] = [(0, 0) for _ in range(len(ex["word"][1:]))]
            cnt = 0
            for h, l in zip(ex["head"][1:], ex["label"][1:]):
                gold_dependencies[i][cnt]=(h - 1, l)
                cnt += 1
        uas,las = evaluate(dependencies, gold_dependencies)  # To check the format of the input, please refer to the utils.py
        print("- test UAS: {:.2f}".format(uas * 100.0), "- test las: {:.2f}".format(las * 100.0))
        print("Done!")
