import torch

from dataloaders.get_loaders import get_loaders
from models.get_models import get_models
from research.Deep_knowledge_tracing_baseline.visualizers.get_visualizers import get_visualizers
from trainers.get_trainers import get_trainers
from utils import get_optimizers, get_crits
from visualizers import get_visualizers

from define_argparser import define_argparser

def main(config):
    #0. device 선언
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    
    #1. 데이터 받아오기
    train_loader, test_loader, num_q = get_loaders(config)
    
    #2. model 선택
    model = get_models(num_q, device, config)
    
    #3. optimizer 선택
    optimizer = get_optimizers(model, config)
    
    #4. criterion 선택
    crit = get_crits(config)
    
    #5. trainer 선택
    trainer = get_trainers(model, optimizer, device, num_q, crit, config)

    #6. 훈련 및 score 계산
    y_true_record, y_score_record = trainer.train(train_loader, test_loader)

    #7. model 기록 저장 위치
    model_path = './train_model_records/' + config.model_fn

    #8. model 기록
    torch.save({
        'model': trainer.model.state_dict(),
        'config': config
    }, model_path)

    #9. 시각화 결과물 만들기
    get_visualizers(y_true_record, y_score_record,model, model_path, test_loader, device, config)

#main
if __name__ == "__main__":
    config = define_argparser() #define_argparser를 불러옴
    main(config)