import torch
from copy import deepcopy

from torch.nn.functional import one_hot
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from random import random, randint

from utils import EarlyStopping

class SAKT_trainer():

    def __init__(
        self,
        model,
        optimizer,
        n_epochs,
        device,
        num_q,
        crit,
        max_seq_len,
        grad_acc=False,
        grad_acc_iter=4
        ):
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.device = device
        self.num_q = num_q
        self.crit = crit
        self.max_seq_len = max_seq_len
        self.grad_acc=grad_acc
        self.grad_acc_iter=grad_acc_iter
    
    def _train(self, train_loader, metric_name):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        for idx, data in enumerate(tqdm(train_loader)):
            self.model.train()
            q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
            q_seqs = q_seqs.to(self.device) #|q_seqs| = (bs, sq) -> [[58., 58., 58., -0., -0., -0., -0., ...], [58., 58., 58., -0., -0., -0., -0., ...]...]
            r_seqs = r_seqs.to(self.device) #|r_seqs| = (bs, sq) -> [[1., 1., 0., -0., -0., -0., -0., ...], [1., 1., 0., -0., -0., -0., -0., ...]...]
            qshft_seqs = qshft_seqs.to(self.device) #|qshft_seqs| = (bs, sq) -> [[58., 58., 58., -0., -0., -0., -0., ...], [58., 58., 58., -0., -0., -0., -0., ...]...]
            rshft_seqs = rshft_seqs.to(self.device) #|rshft_seqs| = (bs, sq) -> [[1., 1., 0., -0., -0., -0., -0., ...], [1., 1., 0., -0., -0., -0., -0., ...]...]
            mask_seqs = mask_seqs.to(self.device) #|mask_seqs| = (bs, sq) -> [[True,  True,  True,  ..., False, False, False], [True,  True,  True,  ..., False, False, False]..]

            #SAKT에서는 결과값이 p와 Mv가 묶인 tuple 형태로 반환되므로, p값만 y_hat으로 넣음
            y_hat, _ = self.model( q_seqs.long(), r_seqs.long(), qshft_seqs.long() ) #|y_hat| = (bs, sq, self.num_q) -> tensor([[[0.6938, 0.7605, ..., 0.7821], [0.8366, 0.6598,  ..., 0.8514],..)
            #=> 각 sq별로 문항의 확률값들이 담긴 벡터들이 나오게 됨

            y_hat = torch.masked_select(y_hat, mask_seqs) #|y_hat| = () -> tensor([0.7782, 0.8887, 0.7638,  ..., 0.8772, 0.8706, 0.8831])
            #=> mask를 활용해서 각 sq 중 실제로 문제를 푼 경우의 확률값만 추출

            correct = torch.masked_select(rshft_seqs, mask_seqs) #|correct| = () -> tensor([0., 1., 1.,  ..., 1., 1., 1.])
            #=> y_hat은 다음 값을 예측하게 되므로, 한칸 뒤의 정답값인 rshft_seqs을 가져옴
            #=> mask를 활용해서 각 sq 중 실제로 문제를 푼 경우의 정답값만 추출

            loss = self.crit(y_hat, correct) #|loss| = () -> ex) 0.5432

            #grad_accumulation
            if self.grad_acc == True:
                loss.backward()
                if (idx + 1) % self.grad_acc_iter == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            y_trues.append(correct)
            y_scores.append(y_hat)
            loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy() #|y_tures| = () -> [0. 0. 0. ... 1. 1. 1.]
        y_scores = torch.cat(y_scores).detach().cpu().numpy() #|y_scores| = () ->  tensor(0.5552)

        auc_score += metrics.roc_auc_score( y_trues, y_scores ) #|metrics.roc_auc_score( y_trues, y_scores )| = () -> 0.6203433289463159

        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        if metric_name == "AUC":
            return auc_score
        elif metric_name == "RMSE":
            return loss_result

    def _validate(self, valid_loader, metric_name):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        with torch.no_grad():
            for data in tqdm(valid_loader):
                self.model.eval()
                q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                qshft_seqs = qshft_seqs.to(self.device)
                rshft_seqs = rshft_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                #SAKT에서는 결과값이 p와 Mv가 묶인 tuple 형태로 반환되므로, p값만 y_hat으로 넣음
                y_hat, _ = self.model( q_seqs.long(), r_seqs.long(), qshft_seqs.long() ) #|y_hat| = (bs, sq, self.num_q) -> tensor([[[0.6938, 0.7605, ..., 0.7821], [0.8366, 0.6598,  ..., 0.8514],..)
                #=> 각 sq별로 문항의 확률값들이 담긴 벡터들이 나오게 됨

                y_hat = torch.masked_select(y_hat, mask_seqs) #|y_hat| = () -> tensor([0.7782, 0.8887, 0.7638,  ..., 0.8772, 0.8706, 0.8831])
                #=> mask를 활용해서 각 sq 중 실제로 문제를 푼 경우의 확률값만 추출

                correct = torch.masked_select(rshft_seqs, mask_seqs) #|correct| = () -> tensor([0., 1., 1.,  ..., 1., 1., 1.])
                #=> y_hat은 다음 값을 예측하게 되므로, 한칸 뒤의 정답값인 rshft_seqs을 가져옴
                #=> mask를 활용해서 각 sq 중 실제로 문제를 푼 경우의 정답값만 추출

                loss = self.crit(y_hat, correct)

                y_trues.append(correct)
                y_scores.append(y_hat)
                loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        if metric_name == "AUC":
            return auc_score
        elif metric_name == "RMSE":
            return loss_result

    def _test(self, test_loader, metric_name):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                self.model.eval()
                q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                qshft_seqs = qshft_seqs.to(self.device)
                rshft_seqs = rshft_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                #SAKT에서는 결과값이 p와 Mv가 묶인 tuple 형태로 반환되므로, p값만 y_hat으로 넣음
                y_hat, _ = self.model( q_seqs.long(), r_seqs.long(), qshft_seqs.long() ) #|y_hat| = (bs, sq, self.num_q) -> tensor([[[0.6938, 0.7605, ..., 0.7821], [0.8366, 0.6598,  ..., 0.8514],..)
                #=> 각 sq별로 문항의 확률값들이 담긴 벡터들이 나오게 됨

                y_hat = torch.masked_select(y_hat, mask_seqs) #|y_hat| = () -> tensor([0.7782, 0.8887, 0.7638,  ..., 0.8772, 0.8706, 0.8831])
                #=> mask를 활용해서 각 sq 중 실제로 문제를 푼 경우의 확률값만 추출

                correct = torch.masked_select(rshft_seqs, mask_seqs) #|correct| = () -> tensor([0., 1., 1.,  ..., 1., 1., 1.])
                #=> y_hat은 다음 값을 예측하게 되므로, 한칸 뒤의 정답값인 rshft_seqs을 가져옴
                #=> mask를 활용해서 각 sq 중 실제로 문제를 푼 경우의 정답값만 추출

                loss = self.crit(y_hat, correct)

                y_trues.append(correct)
                y_scores.append(y_hat)
                loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        if metric_name == "AUC":
            return auc_score
        elif metric_name == "RMSE":
            return loss_result

     #auc용으로 train
    def train(self, train_loader, valid_loader, test_loader, config):
        
        if config.crit == "binary_cross_entropy":
            best_valid_score = 0
            metric_name = "AUC"
        elif config.crit == "rmse":
            best_valid_score = float('inf')
            metric_name = "RMSE"
        test_auc_score = 0
        #출력을 위한 기록용
        train_scores = []
        valid_scores = []
        #best_model = None

        # early_stopping 선언
        early_stopping = EarlyStopping(metric_name=metric_name,
                                    best_score=best_valid_score)

        # Train and Valid Session
        for epoch_index in range(self.n_epochs):
            
            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.n_epochs
            ))

            # Training Session
            train_score = self._train(train_loader, metric_name)
            valid_score = self._validate(valid_loader, metric_name)

            # train, test record 저장
            train_scores.append(train_score)
            valid_scores.append(valid_score)

            # statistics
            train_scores_avg = np.average(train_scores)
            valid_scores_avg = np.average(valid_scores)

            # early stop
            early_stopping(valid_scores_avg, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if config.crit == "binary_cross_entropy":
                if valid_score >= best_valid_score:
                    best_valid_score = valid_score
                    #best_model = deepcopy(self.model.state_dict())
            elif config.crit == "rmse":
                if valid_score <= best_valid_score:
                    best_valid_score = valid_score
                    #best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d) result: train_score=%.4f  valid_score=%.4f  best_valid_score=%.4f" % (
                epoch_index + 1,
                self.n_epochs,
                train_score,
                valid_score,
                best_valid_score,
            ))

        print("\n")
        print("The Best_" + metric_name + "_Score in Training Session is %.4f" % (
                best_valid_score,
            ))
        print("\n")

        # Test Session
        test_auc_score = self._test(test_loader, metric_name)

        print("\n")
        print("The Best_" + metric_name + "_Score in Testing Session is %.4f" % (
                test_auc_score,
            ))
        print("\n")
        
        # 가장 최고의 모델 복구
        #self.model.load_state_dict(best_model)
        self.model.load_state_dict(torch.load("../checkpoints/checkpoint.pt"))

        return train_scores, valid_scores, \
            best_valid_score, test_auc_score