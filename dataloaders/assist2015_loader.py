import os

import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


DATASET_DIR = "datasets/2015_100_skill_builders_main_problems.csv"


class ASSIST2015(Dataset):
    def __init__(self, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        
        self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, \
            self.u2idx = self.preprocess() #가장 아래에서 각각의 요소를 가져옴

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        #match_seq_len은 경우에 따라 설정하기 -> 사용하려면 parameter에 seq_len을 추가해야 함
        #match_seq_len을 거치면, 모든 데이터는 101개로 통일되고, 빈칸인 부분은 -1로 전처리되어있음
        # self.q_seqs, self.r_seqs = \
        #     self.match_seq_len(self.q_seqs, self.r_seqs, seq_len) #아래 method를 한번 거치도록 처리

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        #출력되는 벡터는 모두 101개로 전처리되어있고, 만약 빈칸이 있는 데이터의 경우에는 -1로 채워져있음
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_dir, encoding="ISO-8859-1")
        df = df[(df["correct"] == 0).values + (df["correct"] == 1).values]

        u_list = np.unique(df["user_id"].values) #중복되지 않은 user의 목록
        q_list = np.unique(df["sequence_id"].values) #중복되지 않은 question의 목록

        u2idx = {u: idx for idx, u in enumerate(u_list)} #중복되지 않은 user에게 idx를 붙여준 딕셔너리
        q2idx = {q: idx for idx, q in enumerate(q_list)} #중복되지 않은 question에 idx를 붙여준 딕셔너리

        q_seqs = []
        r_seqs = []

        for u in u_list:
            df_u = df[df["user_id"] == u].sort_values("log_id")

            q_seq = np.array([q2idx[q] for q in df_u["sequence_id"].values])
            r_seq = df_u["correct"].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)

        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx

    # #필요할 경우 사용
    # def match_seq_len(self, q_seqs, r_seqs, seq_len, pad_val=-1):

    #     proc_q_seqs = []
    #     proc_r_seqs = []

    #     for q_seq, r_seq in zip(q_seqs, r_seqs):

    #         #101보다 작은 데이터는 넘기고, 큰 데이터는 while문을 통과하게 됨
    #         #while을 통과할 경우, 101개씩 데이터를 떼서 proc에 넣음
    #         i = 0 #i는 while을 통과할 경우 추가되고, 아니면 추가되지 않음
    #         while i + seq_len + 1 < len(q_seq):
    #             proc_q_seqs.append(q_seq[i:i + seq_len + 1])
    #             proc_r_seqs.append(r_seq[i:i + seq_len + 1])

    #             i += seq_len + 1

    #         #while을 거치지 않은 경우는 바로, while을 거친 경우 남은 데이터에 대해서만 처리하게 됨
    #         proc_q_seqs.append(
    #             np.concatenate(
    #                 [
    #                     q_seq[i:], #while을 거치지 않았다면, 처음부터 끝까지, while을 거쳤다면 남은부분만
    #                     np.array([pad_val] * (i + seq_len + 1 - len(q_seq))) #총 101개로 만들기, 대신 남은 부분은 -1로 채움
    #                 ]
    #             )
    #         )
    #         proc_r_seqs.append(
    #             np.concatenate(
    #                 [
    #                     r_seq[i:],
    #                     np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
    #                 ]
    #             )
    #         )

    #     return proc_q_seqs, proc_r_seqs
