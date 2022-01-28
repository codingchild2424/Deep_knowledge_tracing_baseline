import argparse
import torch

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)

    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.8) #test_ratio는 train_ratio에 따라 정해지도록 설정
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--learning_rate', type=int, default = 0.001)

    #model, opt, dataset, crit 선택
    p.add_argument('--model_name', type=str, default='dkt')
    p.add_argument('--optimizer', type=str, default='adam')
    p.add_argument('--dataset_name', type=str, default = 'assist2015')
    p.add_argument('--crit', type=str, default = 'binary_cross_entropy')

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

    #sakt argument

    #saint argument

    #gkt argument

    
    config = p.parse_args()

    return config