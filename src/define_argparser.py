import argparse
import torch

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)

    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.8) #test_ratio는 train_ratio에 따라 정해지도록 설정
    p.add_argument('--valid_ratio', type=float, default=.1)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--learning_rate', type=int, default = 0.001)

    #model, opt, dataset, crit 선택
    p.add_argument('--model_name', type=str, default='dkt')
    p.add_argument('--optimizer', type=str, default='adam')
    p.add_argument('--dataset_name', type=str, default = 'assist2015')
    p.add_argument('--crit', type=str, default = 'binary_cross_entropy')
    p.add_argument('--max_seq_len', type=int, default=100)

    # grad_accumulation
    p.add_argument('--grad_acc', type=bool, default=False)
    p.add_argument('--grad_acc_iter', type=int, default=4)

    #five_fold cross validation
    p.add_argument('--fivefold', type=bool, default=False)

    #dkt argument
    p.add_argument('--dkt_emb_size', type=int, default = 100)
    p.add_argument('--dkt_hidden_size', type=int, default = 100)

    #dkt+ argument
    p.add_argument('--dkt_plus_emb_size', type=int, default = 100)
    p.add_argument('--dkt_plus_hidden_size', type=int, default = 100)
    p.add_argument('--dkt_plus_lambda_r', type=int, default = 0.01)
    p.add_argument('--dkt_plus_lambda_w1', type=int, default = 0.003)
    p.add_argument('--dkt_plus_lambda_w2', type=int, default = 3.0)

    #dkvmn argument
    p.add_argument('--dkvmn_dim_s', type=int, default = 50)
    p.add_argument('--dkvmn_size_m', type=int, default = 20)
    
    #sakt argument
    p.add_argument('--sakt_n', type=int, default=100)
    p.add_argument('--sakt_d', type=int, default=100)
    p.add_argument('--sakt_num_attn_heads', type=int, default=5)

    #akt


    
    config = p.parse_args()

    return config