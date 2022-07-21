import torch
from torch.nn import Module, Parameter, Embedding, Linear
from torch.nn.init import kaiming_normal_

class DKVMN_c_q(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            dim_s: the dimension of the state vectors in this model
            size_m: the memory size of this model
    '''
    def __init__(self, num_q, num_pid, dim_s, size_m):
        super().__init__()
        self.num_q = num_q
        self.num_pid = num_pid
        self.dim_s = dim_s #default = 50
        self.size_m = size_m #default = 20

        self.pid_emb_layer = Embedding(self.num_pid, self.dim_s)

        self.k_emb_layer = Embedding(self.num_q, self.dim_s) #여기는 q값만 들어오므로, embedding vector의 갯수가 문항수와 동일함
        
        #Mk: Key Matrix로 N개의 Latent Concept(C1, ..., CN)을 인코딩하는 행렬
        #모델의 추론 과정에서 시간과 관계없이 변하지 않는 Static한 메모리
        self.Mk = Parameter(
            torch.Tensor(self.size_m, self.dim_s)
        )

        #Mvt의 초깃값으로 추측됨
        #해당값은 Value Matrix로 (dv, N)의 크기를 가지고 있고
        #N개의 Latent Concept(C1, ..., CN) 각각에 대한 사용자의 Concept State(s1, ..., sN)을 인코딩하고 있음
        #dynamic하게 변화하는 값
        self.Mv0 = Parameter(
            torch.Tensor(self.size_m, self.dim_s)
        )

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_q * 2, self.dim_s) #여기는 (q, r)이 동시에 들어오므로, embedding vector가 두배

        #read 과정에서 사용
        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.p_layer = Linear(self.dim_s, 1)

        #write 과정에서 사용
        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, q, r, pid):

        # 문항과 문제 번호에 따라서 값을 줌
        #self.num_q는 문항의 갯수
        #q는 실제 문항의 번호가 담긴 vectors
        #r은 정오답값
        #|x| = (bs, 문항의 정오답에 따른 인덱스값) ex) (64, 250)
        x = q + self.num_q * r

        #|batch_size| = (64,)
        batch_size = x.shape[0] 

        #|Mv0| = (size_m, dim_s), if dim_s and size_m are default, |Mv0| = (20, 50)
        #|Mvt| = (bs, size_m, dim_s), if dim_s and size_m are default, |Mvt| = (bs, 20, 50)
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        #Mv는 Mvt를 list로 감싸는 코드, 이유는 아래에서 확인해보기
        Mv = [Mvt]

        #k(key)는 q(question)만을 활용해서 embedding vector를 만들게 됨
        #|k| = (bs, 문항 인덱스 값, dim_s), ex) (bs, 250, 50)
        k = self.k_emb_layer(q) + self.pid_emb_layer(pid)

        #|v| = (bs, 문항의 정오답에 따른 인덱스값, dim_s)
        v = self.v_emb_layer(x) + self.pid_emb_layer(pid)

        #|Mk| = (size_m, dim_s)

        #|w| = (bs, 문항 인덱스 값, size_m)
        w = torch.softmax(
            torch.matmul(k, self.Mk.T),
            dim = -1
        )

        #Write Process

        #|e| = (bs, 문항의 정오답에 따른 인덱스값, dim_s)
        e = torch.sigmoid( 
            self.e_layer(v) #|self.e_layer| = (bs, 문항의 정오답에 따른 인덱스값, dim_s)
        )
        #|a| = (bs, 문항의 정오답에 따른 인덱스값, dim_s)
        a = torch.tanh(
            self.a_layer(v) #|self.a_layer| = (bs, 문항의 정오답에 따른 인덱스값, dim_s)
        )

        for et, at, wt in zip(
            e.permute(1, 0, 2), #|e.permute| = (문항의 정오답에 따른 인덱스값, bs, dim_s)
            a.permute(1, 0, 2), #|a.permute| = (문항의 정오답에 따른 인덱스값, bs, dim_s)
            w.permute(1, 0, 2)  #|w.permute| = (문항 인덱스 값, bs, size_m)
        ):
            #|et| = (bs, dim_s)
            #|at| = (bs, dim_s)
            #|wt| = (bs, size_m)

            #형상이 맞지 않는 값들을 더하기 위해 unsqueeze를 활용함
            #|wt.unsqueeze(-1)| = (bs, size_m, 1)
            #|et.unsqueeze(1)| = (bs, 1, dim_s)
            #|at.unsqueeze(1)| = (bs, 1, dim_s)
            #|Mvt| = (bs, size_m, dim_s)
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1)) ) + ( wt.unsqueeze(-1) * at.unsqueeze(1) )
            #Mv는 Mvt들을 보관하는 리스트
            Mv.append(Mvt)

        #Mv는 Mvt들이 쌓여있는 리스트, |Mvt| = (bs, size_m, dim_s)

        Mv = torch.stack(Mv, dim = 1)
        #|Mv| = (bs, Mvt의 갯수, size_m, dim_s)

        #|w.unsqueeze(-1)| = (bs, 문항 인덱스 값, size_m, 1)
        #|Mv[:, :-1]| = (bs, Mvt의 갯수, size_m, dim_s)
        #|(w.unsqueeze(-1) * Mv[:, :-1])| = (bs, 문항 인덱스 값, size_m, dim_s)
        
        #Read Process
        f = torch.tanh(
            self.f_layer( 
                torch.cat(
                    [
                        #rt임, |(w.unsqueeze(-1) * Mv[:, :-1]).sum(-2)| = (bs, 문항 인덱스 값, dim_s)
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        #|k| = (bs, 문항 인덱스 값, dim_s)
                        k
                    ],
                    dim = -1 #마지막 차원을 기준으로 concat
                )
            )
        )

        #|f| = (bs, 문항 인덱스 값, dim_s)

        p = torch.sigmoid( self.p_layer(f) ).squeeze()

        #|p| = (bs, 문항 인덱스별 확률값)

        return p, Mv