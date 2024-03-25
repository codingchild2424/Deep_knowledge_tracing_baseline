import torch
from torch.nn import Module, Parameter, Embedding, \
    Sequential, Linear, ReLU, MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_

class SAKT_c_q_ctt(Module):
    def __init__(self, num_q, num_pid, num_diff, n, d, num_attn_heads, device, dropout=.2): #device를 추가함
        super().__init__()
        self.num_q = num_q #문항의 갯수
        self.num_pid = num_pid
        self.num_diff = 101
        self.n = n #length of the sequence of questions and responses
        self.d = d #dimension of the hidden vectors in this model
        self.num_attn_heads = num_attn_heads #the number of the attention heads in the multi-head attention
        self.dropout = dropout
        self.device = device

        #self.M은 1~(T-1)번째 문제(x_t = (e_t, r_t)) 반응에 대한 embedding이므로, 
        #정답과 오답을 고려하여 문항*2만큼의 embedding vector로 구성됨
        self.M = Embedding(self.num_q * 2, self.d)
        #self.E는 T번째 문제(e_t)에 대한 embedding이므로, 문항 갯수만큼의 embedding vector로 구성됨
        self.E = Embedding(self.num_q, self.d)

        self.emb_pid = Embedding(self.num_pid, self.d)
        self.emb_diff = Embedding(self.num_diff, self.d)

        self.attn = MultiheadAttention(
            self.d, self.num_attn_heads, dropout = self.dropout
        )
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )

        self.FFN_layer_norm = LayerNorm(self.d)

        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry, pid, diff):

        #|q| = (bs, sq)
        #|r| = (bs, sq)
        #|qry| = (bs, sq)

        x = q + self.num_q * r

        M = self.M(x) + self.emb_pid(pid) + self.emb_diff(diff)

        E = self.E(qry) + self.emb_pid(pid) + self.emb_diff(diff)

        M = M.permute(1, 0, 2) #|M| = (sq, bs, d)
        E = E.permute(1, 0, 2) #|E| = (sq, bs, d)

        #형상 때문에 instance로 생성하지 않고 직접 불러옴
        #|P| = (sq, d)
        P = Parameter( 
                torch.Tensor( 
                    M.size(0), #sq로 형상을 맞춤
                    self.d 
                )
            ).to(self.device) #device에 올려줌 / 처음 생성하는 것이 아니면 device가 다름
        kaiming_normal_(P)
        P = P.unsqueeze(1) #|P| = (sq, 1, d)

        #|causal_mask| = (sq, sq)
        causal_mask = torch.triu(
            torch.ones([ E.shape[0], M.shape[0] ]), diagonal = 1
        ).bool().to(self.device)

        M = M + P

        S, attn_weights = self.attn(E, M, M, attn_mask = causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights