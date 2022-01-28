import os

import numpy as np
from paddle import masked_select
import torch

from torch.nn import Module, Embedding, LSTM, Sequential, Linear, Dropout, Sigmoid
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics

class DKT(Module):
    #num_q: 유일한 질문의 갯수
    #emb_size: 100
    #hidden_size: 100
    def __init__(
        self,
        num_q,
        emb_size,
        hidden_size,
        n_layers = 4,
        dropout_p = .2
    ):
        super().__init__()

        self.num_q = num_q #100
        self.emb_size = emb_size #100
        self.hidden_size = hidden_size #100
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        #|self.interaction_emb| = (200, 100)
        self.interaction_emb = Embedding( 
            self.num_q * 2, self.emb_size
        ) 
        #100을 받아서 100이 나옴
        self.lstm_layer = LSTM( 
            input_size = self.emb_size,
            hidden_size = self.hidden_size,
            batch_first = True,
            num_layers = self.n_layers,
            dropout = self.dropout_p
        )  
        #100을 받아서 100이 나옴
        self.out_layer = Sequential(
            Linear(self.hidden_size, self.num_q),
            Sigmoid()
        )

    def forward(self, q, r):
        x = q + self.num_q * r

        z, _ = self.lstm_layer( self.interaction_emb(x) )
        #|z| = (batch, )

        y = self.out_layer(z)

        return y

