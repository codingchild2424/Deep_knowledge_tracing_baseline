import numpy as np
import datetime

import torch
from get_modules.get_loaders import get_loaders
from get_modules.get_models import get_models
from get_modules.get_trainers import get_trainers
from utils import get_optimizers, get_crits, recorder, visualizer

from define_argparser import define_argparser

def main(config, train_loader=None, valid_loader=None, test_loader=None, num_q=None, num_r=None, num_pid=None, num_diff=None):
    #0. device
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    #1. get data
    train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff = get_loaders(config)

    #2. model
    model = get_models(num_q, num_r, num_pid, num_diff, device, config)
    
    #3. optimizer
    optimizer = get_optimizers(model, config)
    
    #4. criterion 선택
    crit = get_crits(config)
    
    #5. trainer
    trainer = get_trainers(model, optimizer, device, num_q, crit, config)

    #6. train
    train_auc_scores, valid_auc_scores, \
        highest_auc_score, test_auc_score  = trainer.train(train_loader, valid_loader, test_loader, config)

    today = datetime.datetime.today()
    record_time = str(today.month) + "_" + str(today.day) + "_" + str(today.hour) + "_" + str(today.minute)

    #7. model 기록 저장 위치
    model_path = '../model_records/' + str(test_auc_score) + "_" + record_time + "_" + config.model_fn

    #8. model 기록
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, model_path)

    return train_auc_scores, valid_auc_scores, highest_auc_score, test_auc_score, record_time

#fivefold main
if __name__ == "__main__":
    config = define_argparser()

    train_auc_scores, valid_auc_scores, \
            best_valid_score, test_auc_score, record_time = main(config)
    recorder(test_auc_score, record_time, config)
    