import torch
from copy import deepcopy

from torch.nn.functional import one_hot
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from random import random, randint

from utils import EarlyStopping

class GKT_trainer():

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
            q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data
            q_seqs = q_seqs.to(self.device)
            r_seqs = r_seqs.to(self.device)
            qshft_seqs = qshft_seqs.to(self.device)
            rshft_seqs = rshft_seqs.to(self.device)
            mask_seqs = mask_seqs.to(self.device)

            y_hat, _ = self.model(q_seqs.long(), r_seqs.long())
            one_hot_vectors = one_hot(qshft_seqs.long(), self.num_q)
            y_hat = (y_hat * one_hot_vectors).sum(-1)

            y_hat = torch.masked_select(y_hat, mask_seqs)
            correct = torch.masked_select(rshft_seqs, mask_seqs)

            loss = self.crit(y_hat, correct)

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

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc = metrics.roc_auc_score(y_trues, y_scores)
        accuracy = metrics.accuracy_score(y_trues, y_scores >= 0.5)

        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        return {
            "loss": loss_result,
            "auc": auc,
            "accuracy": accuracy
        }

    def _validate(self, valid_loader, metric_name):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        with torch.no_grad():
            for data in tqdm(valid_loader):
                self.model.eval()
                q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                qshft_seqs = qshft_seqs.to(self.device)
                rshft_seqs = rshft_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                y_hat, _ = self.model(q_seqs.long(), r_seqs.long())
                y_hat = (y_hat * one_hot(qshft_seqs.long(), self.num_q)).sum(-1)

                y_hat = torch.masked_select(y_hat, mask_seqs)
                correct = torch.masked_select(rshft_seqs, mask_seqs)

                loss = self.crit(y_hat, correct)

                y_trues.append(correct)
                y_scores.append(y_hat)
                loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc = metrics.roc_auc_score(y_trues, y_scores)
        accuracy = metrics.accuracy_score(y_trues, y_scores >= 0.5)

        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        return {
            "loss": loss_result,
            "auc": auc,
            "accuracy": accuracy
        }

    def _test(self, test_loader, metric_name):

        auc_score = 0
        y_trues, y_scores = [], []
        loss_list = []

        with torch.no_grad():
            for data in tqdm(test_loader):
                self.model.eval()
                q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                qshft_seqs = qshft_seqs.to(self.device)
                rshft_seqs = rshft_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                y_hat, _ = self.model(q_seqs.long(), r_seqs.long())
                y_hat = (y_hat * one_hot(qshft_seqs.long(), self.num_q)).sum(-1)

                y_hat = torch.masked_select(y_hat, mask_seqs)
                correct = torch.masked_select(rshft_seqs, mask_seqs)

                loss = self.crit(y_hat, correct)

                y_trues.append(correct)
                y_scores.append(y_hat)
                loss_list.append(loss)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc = metrics.roc_auc_score(y_trues, y_scores)
        accuracy = metrics.accuracy_score(y_trues, y_scores >= 0.5)

        loss_result = torch.mean(torch.Tensor(loss_list)).detach().cpu().numpy()

        return {
            "loss": loss_result,
            "auc": auc,
            "accuracy": accuracy
        }

    def train(self, train_loader, valid_loader, test_loader, config):
        
        if config.crit == "binary_cross_entropy":
            best_valid_auc = 0
            best_test_auc = 0
            best_test_accuracy = 0
            metric_name = "auc"
        elif config.crit == "rmse":
            best_valid_loss = float('inf')
            best_test_loss = float('inf')
            best_test_accuracy = 0
            metric_name = "loss"
        
        train_results = []
        valid_results = []
        test_results = []

        early_stopping = EarlyStopping(metric_name=metric_name,
                                    best_score=best_valid_auc if metric_name == "auc" else best_valid_loss)

        for epoch_index in range(self.n_epochs):
            
            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.n_epochs
            ))

            train_result = self._train(train_loader, metric_name)
            valid_result = self._validate(valid_loader, metric_name)
            test_result = self._test(test_loader, metric_name)

            train_results.append(train_result)
            valid_results.append(valid_result)
            test_results.append(test_result)

            train_metric_avg = np.average([r[metric_name] for r in train_results])
            valid_metric_avg = np.average([r[metric_name] for r in valid_results])
            early_stopping(valid_metric_avg, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if test_result["auc"] >= best_test_auc:
                best_test_auc = test_result["auc"]
            if test_result["accuracy"] >= best_test_accuracy:
                best_test_accuracy = test_result["accuracy"]

            print("Epoch(%d/%d) result: train_loss=%.4f train_auc=%.4f train_accuracy=%.4f valid_loss=%.4f valid_auc=%.4f valid_accuracy=%.4f test_loss=%.4f test_auc=%.4f test_accuracy=%.4f best_test_auc=%.4f best_test_accuracy=%.4f" % (
                epoch_index + 1,
                self.n_epochs,
                train_result["loss"],
                train_result["auc"],
                train_result["accuracy"],
                valid_result["loss"],
                valid_result["auc"],
                valid_result["accuracy"],
                test_result["loss"],
                test_result["auc"],
                test_result["accuracy"],
                best_test_auc,
                best_test_accuracy,
            ))

        print("\n")
        print("The Best Test AUC in Testing Session is %.4f" % (
                best_test_auc,
            ))
        print("The Best Test Accuracy in Testing Session is %.4f" % (
                best_test_accuracy,
            ))
        print("\n")
        
        self.model.load_state_dict(torch.load("../checkpoints/checkpoint.pt"))

        return train_results, valid_results, test_results, \
            best_test_auc, best_test_accuracy