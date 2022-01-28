import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

# if torch.cuda.is_available():
#     from torch.cuda import FloatTensor
#     torch.set_default_tensor_type(torch.cuda.FloatTensor)
# else:
#     from torch import FloatTensor

def collate_fn(batch, pad_val=-1):
    '''
        The collate function for torch.utils.data.DataLoader

        Returns:
            q_seqs: the question(KC) sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            r_seqs: the response sequences with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            qshft_seqs: the question(KC) sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            rshft_seqs: the response sequences which were shifted \
                one step to the right with ths size of \
                [batch_size, maximum_sequence_length_in_the_batch]
            mask_seqs: the mask sequences indicating where \
                the padded entry is with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
    '''
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
    #
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs
