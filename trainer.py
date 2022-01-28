import os

import numpy as np
import torch
from copy import deepcopy

from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics

class Trainer():

    def __init__(self, model, optimizer, num_epochs, device, num_q):
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.num_q = num_q
    
    def _train(self, train_loader):

        auc_score = 0
        y_trues, y_scores = [], []

        for data in train_loader:
            self.model.train()
            q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
            q_seqs = q_seqs.to(self.device)
            r_seqs = r_seqs.to(self.device)
            qshft_seqs = qshft_seqs.to(self.device)
            rshft_seqs = rshft_seqs.to(self.device)
            mask_seqs = mask_seqs.to(self.device)

            y_hat = self.model( q_seqs.long(), r_seqs.long() )
            y_hat = (y_hat * one_hot(qshft_seqs.long(), self.num_q)).sum(-1)

            y_hat = torch.masked_select(y_hat, mask_seqs)
            correct = torch.masked_select(rshft_seqs, mask_seqs)

            self.optimizer.zero_grad()
            loss = binary_cross_entropy(y_hat, correct)
            loss.backward()
            self.optimizer.step()

            y_trues.append(correct)
            y_scores.append(y_hat)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        return auc_score

    def _test(self, test_loader):

        auc_score = 0
        y_trues, y_scores = [], []

        with torch.no_grad():
            for data in test_loader:
                self.model.eval()
                q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                qshft_seqs = qshft_seqs.to(self.device)
                rshft_seqs = rshft_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                y_hat = self.model( q_seqs.long(), r_seqs.long() )
                y_hat = (y_hat * one_hot(qshft_seqs.long(), self.num_q)).sum(-1)

                y_hat = torch.masked_select(y_hat, mask_seqs)
                correct = torch.masked_select(rshft_seqs, mask_seqs)

                y_trues.append(correct)
                y_scores.append(y_hat)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        return auc_score

    def train(self, train_loader, test_loader):
        
        highest_auc_score = 0
        best_model = None

        for epoch_index in range(self.num_epochs):
            
            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.num_epochs
            ))

            train_auc_score = self._train(train_loader)
            test_auc_score = self._test(test_loader)

            if test_auc_score >= highest_auc_score:
                highest_auc_score = test_auc_score
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d) result: train_auc_score=%.4f  test_auc_score=%.4f  highest_auc_score=%.4f" % (
                epoch_index + 1,
                self.num_epochs,
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