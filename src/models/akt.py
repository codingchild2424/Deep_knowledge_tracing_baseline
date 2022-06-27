from matplotlib.pyplot import new_figure_manager
import numpy as np
import torch.nn as nn
import torch

class MonotonicAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self):

        minus_theta = -1e8
        distance = self.distance_func()

        np.exp(minus_theta * distance)

        
        s = 0

        a = self.softmax(s)

        return a

    #grad를 갱신하지 않음
    @torch.no_grad()
    def distance_func(self, t, tau):

        

        np.absolute(t - tau)

class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass

# 원 코드 69~93까지
class RaschModelEmbedding(nn.Module):
    def __init__(
        self,
        num_q, #num_q: concept의 unique한 수
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
        self.qr_emb = nn.EmbeddingBag(self.num_q * 2, self.emb_size)
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

        pass

    def forward(self, q, r, pid):
        
        # rasch_emb
        x_t, y_t, c_reg_loss = self.rasch_emb(q, r, pid)