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
        ):
        super().__init__()

        self.emb_size = emb_size

        # c_ct
        self.q_emb = nn.Embedding(self.num_q, self.emb_size)
        # (ct, rt)
        self.qr_emb = nn.EmbeddingBag(self.num_q * 2, self.emb_size)
        # d_ct
        self.q_emb_diff = nn.Embedding(self.num_q, self.emb_size)
        # difficult parameter
        self.u_emb = nn.Embedding(self.num_pid, 1)


    def forward(self, q, qr, pid):
        """
        q_data: concept
        qr: concept + response
        pid: problem id
        """
        q_emb = self.q_emb(q)
        qr_emb = self.qr_emb(qr)


        pass
        


# AKT Main 모델
class AKT(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass