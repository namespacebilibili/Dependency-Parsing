import torch
import torch.nn as nn
from utils import evaluate
from parser_utils import *
import random
import numpy as np
class ParserTrainer():

    def __init__(
        self,
        train_data,
        dev_data,
        optimizer,
        loss_func,
        output_path,
        batch_size=1024,
        n_epochs=10,
        lr=0.0005, 
        weight_decay = 1e-8
    ): # You can add more arguments
        """
        Initialize the trainer.
        
        Inputs:
            - train_data: Packed train data
            - dev_data: Packed dev data
            - optimizer: The optimizer used to optimize the parsing model
            - loss_func: The cross entropy function to calculate loss, initialized beforehand
            - output_path (str): Path to which model weights and results are written
            - batch_size (int): Number of examples in a single batch
            - n_epochs (int): Number of training epochs
            - lr (float): Learning rate
        """
        self.train_data = train_data
        self.dev_data = dev_data
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.output_path = output_path
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ### TODO: You can add more initializations here

        self.config = Config()



    def train(self, parser, ): # You can add more arguments as you need
        """
        Given packed train_data, train the neural dependency parser (including optimization),
        save checkpoints, print loss, log the best epoch, and run tests on packed dev_data.

        Inputs:
            - parser (Parser): Neural Dependency Parser
        """
        best_dev_UAS = 0

        ### TODO: Initialize `self.optimizer`, i.e., specify parameters to optimize
        self.optimizer = torch.optim.Adam(parser.model.parameters(), lr=self.lr, weight_decay=1e-8)
        for epoch in range(self.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.n_epochs))
            dev_UAS, dev_LAS = self._train_for_epoch(parser, )
            # TODO: you can change this part, to use either uas or las to select best model
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                print("New best dev UAS! Saving model.")
                torch.save(parser.model.state_dict(), self.output_path)
            print("")


    def _train_for_epoch(self, parser, ): # You can add more arguments as you need
        """ 
        Train the neural dependency parser for single epoch.

        Inputs:
            - parser (Parser): Neural Dependency Parser
        Return:
            - dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
        """
        def get_batches(data, batch_size):
            def _batch(data, minibatch_idx):
                return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]

            x = np.array([d[0] for d in data])
            y = np.array([d[2] for d in data])
            # deprel = np.array([d[1] for d in data])
            label = np.zeros((y.size,parser.n_trans))
            label[np.arange(y.size), y] = 1
            data = [x, label]
            list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
            data_size = len(data[0]) if list_data else len(data)
            indices = np.arange(data_size)
            np.random.shuffle(indices)
            for batch_start in np.arange(0, data_size, batch_size):
                minibatch_indices = indices[batch_start:batch_start + batch_size]
                yield [_batch(d, minibatch_indices) for d in data] if list_data \
                    else _batch(data, minibatch_indices)

        parser.model.train() # Places model in "train" mode, e.g., apply dropout layer, etc.
        ### TODO: Train all batches of train_data in an epoch.
        ### Remember to shuffle before training the first batch (You can use Dataloader of PyTorch)
        
        total_loss = 0
        count = 0
        for i, (x, y) in enumerate(get_batches(self.train_data, self.batch_size)):
            self.optimizer.zero_grad()
            loss_trans = 0
            loss_label = 0
            x = torch.from_numpy(x).long().to(self.device)
            y = torch.from_numpy(y.nonzero()[1]).long().to(self.device)
            # z = torch.from_numpy(z.nonzero()[1]).long().to(self.device)
            predict = parser.model.forward(x)
            loss = self.loss_func(predict, y)
            # loss_label = self.loss_func(label, z)
            # loss = loss_trans + loss_label
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            count += 1
        print("average batch loss: ", total_loss/count)
       
        print("Evaluating on dev set",)
        parser.model.eval() # Places model in "eval" mode, e.g., don't apply dropout layer, etc.
        UAS, dependencies = parser.parse(self.dev_data)
        temp = [[] for _ in range(len(dependencies))]
        idx = 0
        print(self.dev_data[0])
        print(dependencies[0])
        print(len(dependencies[0]),len(self.dev_data[0]["head"][1:]))
        for dependency in dependencies:
            temp[idx] = [(0, 0) for _ in range(len(dependency))]
            for (x, y, z) in dependency:
                # print(len(temp[idx]), y - 1)
                temp[idx][y - 1] = (x - 1, parser.tok2id[parser.config.L_PREFIX + z])
            idx += 1
        dependencies = temp
        gold_dependencies = [[] for _ in range(len(self.dev_data))]
        for i, ex in enumerate(self.dev_data):
            gold_dependencies[i] = [(0, 0) for _ in range(len(ex["word"][1:]))]
            cnt = 0
            for h, l in zip(ex["head"][1:], ex["label"][1:]):
                gold_dependencies[i][cnt]=(h - 1, l)
                cnt += 1
        uas,las = evaluate(dependencies, gold_dependencies)  # To check the format of the input, please refer to the utils.py

        ### NOTE: BUGGY CODE HERE
        print("- dev UAS: {:.2f}".format(uas * 100.0), "- dev LAS: {:.2f}".format(las * 100.0))
        return UAS, 0
