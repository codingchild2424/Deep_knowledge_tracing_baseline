from matplotlib.pyplot import new_figure_manager
import torch.nn as nn
from torch.nn import Module, LSTM, Sequential, Linear, Sigmoid

class DKT_c_rasch(Module):
    #num_q: 유일한 질문의 갯수
    #emb_size: 100
    #hidden_size: 100
    def __init__(
        self,
        num_q,
        num_r,
        num_pid,
        emb_size,
        hidden_size,
        n_layers = 4,
        dropout_p = .2
    ):
        super().__init__()

        self.num_q = num_q #100

        # 추가
        self.num_r = num_r
        self.num_pid = num_pid

        self.emb_size = emb_size #100
        self.hidden_size = hidden_size #100
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # #|self.interaction_emb| = (200, 100) -> 즉, 전체 문항의 맞고 틀림을 고려해서 문항수*2만큼의 행이 만들어지고, 각 행들은 embedding값으로 채워짐
        # self.interaction_emb = Embedding( 
        #     self.num_q * 2, self.emb_size
        # ) 

        self.emb_q = nn.Embedding(self.num_q, self.emb_size)
        self.emb_r = nn.Embedding(self.num_r, self.emb_size) # 0 or 1
        self.emb_pid = nn.Embedding(self.num_pid, 1)


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

    def forward(self, q_seqs, r_seqs, pid_seqs):
        #|q_seqs| = |r_seqs| = |pid_seqs| = (bs, sq)

        rasch_emb = self.emb_pid(pid_seqs) * self.emb_q(q_seqs)

        emb = self.emb_q(q_seqs) + self.emb_r(r_seqs) + rasch_emb #|x| = (bs, sq)

        z, _ = self.lstm_layer( emb ) #|z| = (bs, sq, self.hidden_size)

        y = self.out_layer(z) #|y| = (bs, sq, self.num_q) -> 통과시키면 확률값이 나옴

        return y

