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

class RaschModelEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self):
        pass


# AKT Main 모델
class AKT(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self):
        pass