import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from torch.nn.functional import binary_cross_entropy

from dataloaders.assist2015_loader import ASSIST2015

from models.dkt import DKT
from models.dkt_plus import DKT_plus

from utils import collate_fn
from trainers.dkt_trainer import DKT_trainer
from trainers.dkt_plus_trainer import DKT_plus_trainer
from define_argparser import define_argparser

from visualizers.roc_auc_visualizer import roc_curve_visualizer
from visualizers.personal_pred_visualizer import personal_pred_visualizer

def main(config):
    #device 선언
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    #1. dataset 선택
    if config.dataset_name == "assist2015":
        dataset = ASSIST2015()
    #-> 추가적인 데이터셋
    else:
        print("Wrong dataset_name was used...")

    #2. model 선택
    if config.model_name == "dkt":
        model = DKT(
            num_q = dataset.num_q,
            emb_size = config.dkt_emb_size,
            hidden_size = config.dkt_hidden_size
        ).to(device)
    elif config.model_name == "dkt_plus":
        model = DKT_plus(
            num_q = dataset.num_q,
            emb_size = config.dkt_plus_emb_size,
            hidden_size = config.dkt_plus_hidden_size,
        ).to(device)
    #-> 추가적인 모델 정의
    else:
        print("Wrong model_name was used...")

    #3. optimizer 선택
    if config.optimizer == "adam":
        optimizer = Adam(model.parameters(), config.learning_rate)
    elif config.optimizer == "SGD":
        optimizer = SGD(model.parameters(), config.learning_rate)
    #-> 추가적인 optimizer 설정
    else:
        print("Wrong optimizer was used...")

    #4. criterion 선택
    if config.crit == "binary_cross_entropy":
        crit = binary_cross_entropy
    #-> 추가적인 criterion 설정
    else:
        print("Wrong criterion was used...")

    #train, test 데이터 나누기
    train_size = int( len(dataset) * config.train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [ train_size, test_size ]
    )

    #train, test 데이터 섞기
    train_loader = DataLoader(
        train_dataset, batch_size = config.batch_size, shuffle = True,
        collate_fn = collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size = config.batch_size, shuffle = True,
        collate_fn = collate_fn
    )

    #trainer 실행
    if config.model_name == "dkt":
        trainer = DKT_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = dataset.num_q,
            crit = crit
        )
        y_ture_record, y_score_record = \
            trainer.train(train_loader, test_loader)
    elif config.model_name == "dkt_plus":
        trainer = DKT_plus_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = dataset.num_q,
            crit = crit,
            lambda_r = config.dkt_plus_lambda_r,
            lambda_w1 = config.dkt_plus_lambda_w1,
            lambda_w2 = config.dkt_plus_lambda_w2
        )
        y_ture_record, y_score_record = \
            trainer.train(train_loader, test_loader)
    #=> 새로운 trainer 넣기

    #model 기록 저장 위치
    model_path = './train_model_records/' + config.model_fn

    #model 기록
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, model_path)

    #시각화 결과물 만들기
    if config.model_name == "dkt":
        roc_curve_visualizer(y_ture_record, y_score_record, config.model_name)
        personal_pred_visualizer(model, model_path, test_loader, device, config.model_name , dataset.num_q)
    elif config.model_name == "dkt_plus":
        roc_curve_visualizer(y_ture_record, y_score_record, config.model_name)

#main
if __name__ == "__main__":
    config = define_argparser() #define_argparser를 불러옴
    main(config)