from paddle import masked_select
import torch
from copy import deepcopy

from torch.nn.functional import one_hot
from sklearn import metrics

from tqdm import tqdm

class DKT_plus_trainer():

    def __init__(self, model, optimizer, n_epochs, device, num_q, crit, lambda_r, lambda_w1, lambda_w2):
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.device = device
        self.num_q = num_q
        self.crit = crit
        self.lambda_r = lambda_r
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2

        print(self.model)
    
    def _train(self, train_loader):

        auc_score = 0
        y_trues, y_scores = [], []

        for data in tqdm(train_loader):
            self.model.train()
            q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
            q_seqs = q_seqs.to(self.device) #|q_seqs| = (bs, sq) -> [[58., 58., 58., -0., -0., -0., -0., ...], [58., 58., 58., -0., -0., -0., -0., ...]...]
            r_seqs = r_seqs.to(self.device) #|r_seqs| = (bs, sq) -> [[1., 1., 0., -0., -0., -0., -0., ...], [1., 1., 0., -0., -0., -0., -0., ...]...]
            qshft_seqs = qshft_seqs.to(self.device) #|qshft_seqs| = (bs, sq) -> [[58., 58., 58., -0., -0., -0., -0., ...], [58., 58., 58., -0., -0., -0., -0., ...]...]
            rshft_seqs = rshft_seqs.to(self.device) #|rshft_seqs| = (bs, sq) -> [[1., 1., 0., -0., -0., -0., -0., ...], [1., 1., 0., -0., -0., -0., -0., ...]...]
            mask_seqs = mask_seqs.to(self.device) #|mask_seqs| = (bs, sq) -> [[True,  True,  True,  ..., False, False, False], [True,  True,  True,  ..., False, False, False]..]

            y_hat = self.model( q_seqs.long(), r_seqs.long() ) #|y_hat| = (bs, sq, self.num_q) -> tensor([[[0.6938, 0.7605, ..., 0.7821], [0.8366, 0.6598,  ..., 0.8514],..)
            #=> 각 sq별로 문항의 확률값들이 담긴 벡터들이 나오게 됨

            #|one_hot_vectors| = (bs, sq, self.num_q)
            y_curr = ( y_hat * one_hot(q_seqs.long(), self.num_q) ).sum(-1) #|y_curr| = (bs, sq) -> 현재 값에 대한 확률, 각 sq에는 그 문항에 해당하는 확률값만 담기게 됨
            y_next = ( y_hat * one_hot(qshft_seqs.long(), self.num_q) ).sum(-1) #|y_next| = (bs, sq) -> 다음 값에 대한 확률, 각 sq에는 그 문항에 해당하는 확률값만 담기게 됨

            y_curr = torch.masked_select(y_curr, mask_seqs) #|y_curr| = () -> 현재 확률값
            y_next = torch.masked_select(y_next, mask_seqs) #|y_curr| = () -> 다음 확률값, 원래 DKT는 이것만 봄
            correct = torch.masked_select(r_seqs, mask_seqs) #|correct| = () -> 현재 정답값
            correct_shft = torch.masked_select(rshft_seqs, mask_seqs) #|correct_shft| = () -> 다음 정답값

            loss_w1 = torch.masked_select(
                torch.norm(y_hat[:, 1:] - y_hat[:, :-1], p = 1, dim = -1),
                mask_seqs[:, 1:]
            )
            loss_w2 = torch.masked_select(
                (torch.norm(y_hat[:, 1:] - y_hat[:, :-1], p = 2, dim = -1) ** 2),
                mask_seqs[:, 1:]
            )

            self.optimizer.zero_grad()
            #self.crit은 binary_cross_entropy
            loss = \
                self.crit(y_next, correct_shft) + \
                self.lambda_r * self.crit(y_curr, correct) + \
                self.lambda_w1 * loss_w1.mean() / self.num_q + \
                self.lambda_w2 * loss_w2.mean() / self.num_q

            loss.backward()
            self.optimizer.step()

            y_trues.append(correct_shft)
            y_scores.append(y_next)

        y_trues = torch.cat(y_trues).detach().cpu().numpy() #|y_tures| = () -> [0. 0. 0. ... 1. 1. 1.]
        y_scores = torch.cat(y_scores).detach().cpu().numpy() #|y_scores| = () ->  tensor(0.5552)

        auc_score += metrics.roc_auc_score( y_trues, y_scores ) #|metrics.roc_auc_score( y_trues, y_scores )| = () -> 0.6203433289463159

        return auc_score

    def _test(self, test_loader):

        auc_score = 0
        y_trues, y_scores = [], []

        with torch.no_grad():
            for data in tqdm(test_loader):
                self.model.eval()
                q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                qshft_seqs = qshft_seqs.to(self.device)
                rshft_seqs = rshft_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                y_hat = self.model( q_seqs.long(), r_seqs.long() )
                y_next = (y_hat * one_hot(qshft_seqs.long(), self.num_q)).sum(-1)

                y_next = torch.masked_select(y_next, mask_seqs)
                correct_shft = torch.masked_select(rshft_seqs, mask_seqs)

                y_trues.append(correct_shft)
                y_scores.append(y_next)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        return auc_score

    def train(self, train_loader, test_loader):
        
        highest_auc_score = 0
        best_model = None

        for epoch_index in range(self.n_epochs):
            
            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.n_epochs
            ))

            train_auc_score = self._train(train_loader)
            test_auc_score = self._test(test_loader)

            if test_auc_score >= highest_auc_score:
                highest_auc_score = test_auc_score
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d) result: train_auc_score=%.4f  test_auc_score=%.4f  highest_auc_score=%.4f" % (
                epoch_index + 1,
                self.n_epochs,
                train_auc_score,
                test_auc_score,
                highest_auc_score,
            ))

        print("\n")
        print("The Highest_Auc_Score in Training Session is %.4f" % (
                highest_auc_score,
            ))
        print("\n")
        
        # 가장 최고의 모델 복구    
        self.model.load_state_dict(best_model)