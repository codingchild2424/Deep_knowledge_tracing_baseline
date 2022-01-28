import os
import argparse
import json
import pickle

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from dataloaders.assist2015_loader import ASSIST2015

from models.dkt import DKT
from utils import collate_fn
from trainer import Trainer

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8) #test_ratio는 train_ratio에 따라 정해지도록 설정

    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--model_name', type=str, default='dkt')
    p.add_argument('--optimizer', type=str, default='adam')

    p.add_argument('--dataset_name', type=str, default = 'assist2015')

    p.add_argument('--learning_rate', type=int, default = 0.001)

    config = p.parse_args()

    return config

def main(config):
    #device 선언
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    #batch_size = config.batch_size
    #num_epochs = config.n_epochs
    #train_ratio = config.train_ratio
    #optimizer = config.optimizer
    #model_name = config.model_name
    #dataset_name = config.dataset_name
    #learning_rate = config.learning_rate

    if config.dataset_name == "assist2015":
        dataset = ASSIST2015()
    #-> 추가적인 데이터셋
    else:
        print("Wrong dataset_name was used...")

    if config.model_name == "dkt":
        #일단 하드코딩했는데, 나중에 config로 전달해보기
        model = DKT(dataset.num_q, emb_size = 100, hidden_size = 100).to(device) 
    #-> 추가적인 모델 정의
    else:
        print("Wrong model_name was used...")

    if config.optimizer == "adam":
        optimizer = Adam(model.parameters(), config.learning_rate)

    train_size = int( len(dataset) * config.train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [ train_size, test_size ]
    )

    train_loader = DataLoader(
        train_dataset, batch_size = config.batch_size, shuffle = True,
        collate_fn = collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size = config.batch_size, shuffle = True,
        collate_fn = collate_fn
    )

    trainer = Trainer(model, optimizer, config.n_epochs, device, dataset.num_q)

    trainer.train(train_loader, test_loader)

    #model 기록
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, config.model_fn)

if __name__ == "__main__":
    config = define_argparser()
    main(config)