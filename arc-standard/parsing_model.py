import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import parser_utils

class CubeActivation(nn.Module):
    def __init__(self):
        super(CubeActivation, self).__init__()

    def forward(self, x):
        return torch.pow(x, 3)

class ParsingModel(nn.Module):
    def __init__(self, embeddings, out_size, config:parser_utils.Config):
        """ 
        Initialize the parser model. You can add arguments/settings as you want, depending on how you design your model.
        NOTE: You can load some pretrained embeddings here (If you are using any).
              Of course, if you are not planning to use pretrained embeddings, you don't need to do this.
        """
        super(ParsingModel, self).__init__()
        self.hidden_size = config.hidden_size
        self.embedding_size = embeddings.shape[1]
        self.vocab_size = config.vocab_size
        self.vocab = config.vocab
        self.out_size = out_size
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.dropout = nn.Dropout(config.dropout)
        self.cube = CubeActivation()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.relu = nn.ReLU()
        word_vec = {}
        for line in open(config.vector_file):
            item = line.strip().split()
            word_vec[item[0]] = [float(i) for i in item[1:]]
        self.word_embeddings.weight = nn.Parameter(torch.from_numpy(embeddings).to(self.device))
        if config.use_dep:
            self.hidden_layer = nn.Linear(self.embedding_size * 48, self.hidden_size, bias=True)
        else:
            self.hidden_layer = nn.Linear(self.embedding_size * 36, self.hidden_size, bias=True)
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.hidden_size // 2, bias = True)
        self.output_layer = nn.Linear(self.hidden_size // 2, out_size, bias = True)
        nn.init.xavier_uniform_(self.hidden_layer.weight, gain = 1)
        # nn.init.xavier_uniform_(self.hidden_layer2.weight, gain = 1)

    def forward(self, t):
        """
        Input: input tensor of tokens -> SHAPE (batch_size, n_features)
        Return: tensor of predictions (output after applying the layers of the network
                                 without applying softmax) -> SHAPE (batch_size, n_classes)
        """
        t = t.to(self.device)
        x = self.word_embeddings(t)
        x = x.view(x.size()[0], -1)
        x = self.dropout(x)
        x = self.hidden_layer(x)
        # x = self.cube(x)
        x = self.relu(x)
        x= self.hidden_layer2(x)
        x = self.relu(x)
        logits = self.output_layer(x)

        return logits
