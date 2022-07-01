import pandas as pd
import numpy as np
import csv

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from torch.optim import SGD, Adam

from torch.nn.functional import binary_cross_entropy

import matplotlib.pyplot as plt

def collate_fn(batch, pad_val=-1):

    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []

    for q_seq, r_seq in batch:
        q_seqs.append(torch.Tensor(q_seq[:-1])) #총 데이터(M개) 중에서 앞의 첫번째 ~ (M-1), 갯수 M-1개
        r_seqs.append(torch.Tensor(r_seq[:-1])) #총 데이터(M개) 중에서 앞의 첫번째 ~ (M-1), 갯수 M-1개
        qshft_seqs.append(torch.Tensor(q_seq[1:])) #총 데이터(M개) 중에서 앞의 두번째 ~ M, 갯수 M-1개
        rshft_seqs.append(torch.Tensor(r_seq[1:])) #총 데이터(M개) 중에서 앞의 두번째 ~ M, 갯수 M-1개

    #가장 길이가 긴 seqs를 기준으로 길이를 맞추고, 길이를 맞추기 위해 그 자리에는 -1(pad_val)을 넣어줌
    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )

    #각 원소가 -1이 아니면 Ture, -1이면 False로 값을 채움
    #이후 (q_seqs != pad_val)과 (qshft_seqs != pad_val)을 곱해줌 => 그러면 qshft가 -1이 하나 더 많을 것이므로, qshft 기준으로 True 갯수가 맞춰짐
    #mask_seqs는 실제로 문항이 있는 경우만을 추출하기 위해 사용됨(실제 문항이 있다면, True, 아니면 False, pad_val은 전체 길이를 맞춰주기 위해 사용됨)
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    #즉 전체를 qshft_seqs의 -1이 아닌 갯수만큼은 true(1)을 곱해서 원래 값을 부여하고, 아닌 것은 False(0)을 곱해서 0으로 만듦
    q_seqs, r_seqs, qshft_seqs, rshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs
    #|q_seqs| = (batch_size, maximum_sequence_length_in_the_batch)
    #|r_seqs| = (batch_size, maximum_sequence_length_in_the_batch)
    #|qshft_seqs| = (batch_size, maximum_sequence_length_in_the_batch)
    #|rshft_seqs| = (batch_size, maximum_sequence_length_in_the_batch)
    #|mask_seqs| = (batch_size, maximum_sequence_length_in_the_batch)

def pid_collate_fn(batch, pad_val=-1):

    q_seqs = []
    r_seqs = []
    pid_seqs = []

    for q_seq, r_seq, pid_seq in batch:

        q_seqs.append(torch.Tensor(q_seq)) #총 데이터(M개) 중에서 앞의 첫번째 ~ (M-1), 갯수 M-1개
        r_seqs.append(torch.Tensor(r_seq)) #총 데이터(M개) 중에서 앞의 첫번째 ~ (M-1), 갯수 M-1개
        pid_seqs.append(torch.Tensor(pid_seq)) #총 데이터(M개) 중에서 앞의 첫번째 ~ (M-1), 갯수 M-1개

    #가장 길이가 긴 seqs를 기준으로 길이를 맞추고, 길이를 맞추기 위해 그 자리에는 -1(pad_val)을 넣어줌
    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    pid_seqs = pad_sequence(
        pid_seqs, batch_first=True, padding_value=pad_val
    )

    #각 원소가 -1이 아니면 Ture, -1이면 False로 값을 채움
    #mask_seqs는 실제로 문항이 있는 경우만을 추출하기 위해 사용됨(실제 문항이 있다면, True, 아니면 False, pad_val은 전체 길이를 맞춰주기 위해 사용됨)
    mask_seqs = (q_seqs != pad_val)

    #즉 전체를 qshft_seqs의 -1이 아닌 갯수만큼은 true(1)을 곱해서 원래 값을 부여하고, 아닌 것은 False(0)을 곱해서 0으로 만듦
    q_seqs, r_seqs, pid_seqs = q_seqs * mask_seqs, r_seqs * mask_seqs, pid_seqs * mask_seqs

    return q_seqs, r_seqs, pid_seqs, mask_seqs
    #|q_seqs| = (batch_size, maximum_sequence_length_in_the_batch)
    #|r_seqs| = (batch_size, maximum_sequence_length_in_the_batch)
    #|mask_seqs| = (batch_size, maximum_sequence_length_in_the_batch)


#get_optimizer 정의
def get_optimizers(model, config):
    if config.optimizer == "adam":
        optimizer = Adam(model.parameters(), config.learning_rate)
    elif config.optimizer == "SGD":
        optimizer = SGD(model.parameters(), config.learning_rate)
    #-> 추가적인 optimizer 설정
    else:
        print("Wrong optimizer was used...")

    return optimizer

#get_crit 정의
#get_crit 정의
def get_crits(config):
    if config.crit == "binary_cross_entropy":
        crit = binary_cross_entropy
    #-> 추가적인 criterion 설정
    elif config.crit == "rmse":
        class RMSELoss(nn.Module):
            def __init__(self, eps=1e-8):
                super().__init__()
                self.mse = nn.MSELoss()
                self.eps = eps

            def forward(self, y_hat, y):
                loss =  torch.sqrt(self.mse(y_hat, y) + self.eps)
                return loss

        crit = RMSELoss()
    else:
        print("Wrong criterion was used...")

    return crit

# early stop
class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, metric_name, best_score=0, patience=10, verbose=True, delta=0, path='../checkpoints/checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 10
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.metric_name = metric_name
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = best_score
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = val_loss
        
        # AUC 값이 커져야 성능 증가
        if self.metric_name == "AUC":
            # best_score가 없을 때
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            # 현재 값이 best_score보다 작을때(성능 감소)
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            # 현재 값이 best_score보다 커질때(성능 증가)
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        # RMSE 값이 줄어야 성능 증가
        elif self.metric_name == "RMSE":
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            # 현재 값이 best_score보다 클때(성능 감소)
            elif score > self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            # 현재 값이 best_score보다 작아질때(성능 증가)
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss was updated ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

#recoder
#깔끔하게 한장의 csv로 나오도록 바꿔보기
def recorder(test_score, record_time, config):

    dir_path = "../score_records/"
    record_path = dir_path + "auc_record.csv"

    #리스트에 모든 값 더해서 1줄로 만들기
    append_list = []

    append_list.append(record_time)
    append_list.extend([
        config.model_fn, config.model_name, config.dataset_name
    ])
    append_list.append(config.crit + "_test_score")
    append_list.append(test_score)

    #csv파일 열어서 한줄 추가해주기
    with open(record_path, 'a', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(append_list)

# visualizer도 여기에 하나 만들기
def visualizer(train_auc_scores, valid_auc_scores, record_time):
    plt.plot(train_auc_scores)
    plt.plot(valid_auc_scores)
    plt.legend(['train_auc_scores', 'valid_auc_scores'])
    path = "../graphs/"
    plt.savefig(path + record_time + ".png")