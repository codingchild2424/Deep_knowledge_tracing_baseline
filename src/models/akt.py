from matplotlib.pyplot import new_figure_manager
import numpy as np
import torch.nn as nn
import torch
import math

class MonotonicAttention(nn.Module):
    def __init__(
        self,
        dropout_p,
        ):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, Q, K, V, mask=None, dk=64, gamma=None):
        # |Q| = (batch_size, m, hidden_size)
        # |K| = |V| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, m, n)

        # w = attention energy
        w = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(dk)
        # |w| = (batch_size, m, n)

        if mask is not None:
            assert w.size() == mask.size()
            # mask를 -float('inf')로 만들어두니 overflow 문제 발생
            w.masked_fill_(mask, -1e8)
        
        distance_score = self.distance_func(w)
        minus_theta = -1e8

        s = np.exp(minus_theta * distance_score) * w

        a = self.softmax(s)

        return a

    #grad를 갱신하지 않음, 원 코드 304라인부터 확인
    @torch.no_grad()
    def distance_func(self, w):

        score = self.softmax(w) # gamma_t_t'

        tau = torch.cumsum(score, dim=-1)
        gamma = torch.sum(score, dim=-1, deepdim=True)

        position_effect = torch.abs(tau - gamma)

        return 0



        

# class Attention(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, Q, K, V, mask=None, dk=64):
#         # |Q| = (batch_size, m, hidden_size)
#         # |K| = |V| = (batch_size, n, hidden_size)
#         # |mask| = (batch_size, m, n)

#         # w = attention energy
#         w = torch.bmm(Q, K.transpose(1, 2))

#         # |w| = (batch_size, m, n)
#         if mask is not None:
#             assert w.size() == mask.size()
#             # mask를 -float('inf')로 만들어두니 overflow 문제 발생
#             w.masked_fill_(mask, -1e8)

#         w = self.softmax(w / (dk**.5)) #attention값
#         c = torch.bmm(w, V) #attention값과 Value값 행렬곱
#         # |c| = (batch_size, m, hidden_size)

#         return c



class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass

class QKEncoder(nn.Module):
    def __init__(
        self,
        num_q,
        n_heads,
        dropout,
        d_model,
        d_feature,
        d_ff,
        kq_same,
        model_type
        ):
        super().__init__()
        pass
    def forward(self):
        pass

class Retriever(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass

class RaschModelEmbedding(nn.Module):
    def __init__(
        self,
        emb_size,
        l2=1e-5,
        ):
        super().__init__()

        self.emb_size = emb_size
        self.l2 = l2
        # c_ct
        self.q_emb = nn.Embedding(self.num_q, self.emb_size)
        # d_ct
        self.q_emb_diff = nn.Embedding(self.num_q, self.emb_size)
        # e_(ct, rt)
        self.qr_emb = nn.Embedding(self.num_q * 2, self.emb_size)
        # f_(ct, rt)
        self.qr_emb_diff = nn.Embedding(self.num_q * 2, self.emb_size)
        # u, difficult parameter
        self.diff_emb = nn.Embedding(self.num_pid, 1)

    def forward(self, q, qr, pid):
        """
        q_data: concept
        qr: concept + response
        pid: problem id
        """
        q_emb = self.q_emb(q)#c_c_t
        q_emb_diff = self.q_emb_diff(q) #d_c_t

        qr_emb = self.qr_emb(qr)
        qr_emb_diff = self.qr_emb_diff(qr)

        diff_emb = self.diff_emb(pid) #u_q_t

        x_t = q_emb + diff_emb * q_emb_diff
        y_t = qr_emb + diff_emb * qr_emb_diff

        # 이게 어디에 쓰이는지는 모르겠음
        c_reg_loss = (diff_emb ** 2.).sum() * self.l2

        return x_t, y_t, c_reg_loss

# AKT Main 모델
class AKT(nn.Module):

    def __init__(self, num_q, emb_size):
        super().__init__()

        self.num_q = num_q
        self.emb_size = emb_size

        self.rasch_emb = RaschModelEmbedding(self.num_q, self.emb_size)

    def forward(self, q, r, pid):
        
        # rasch_emb
        x_t, y_t, c_reg_loss = self.rasch_emb(q, r, pid)