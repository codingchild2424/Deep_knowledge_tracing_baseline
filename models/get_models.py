from models.dkt import DKT
from models.dkt_plus import DKT_plus

def get_models(num_q, device, config):

    if config.model_name == "dkt":
        model = DKT(
            num_q = num_q,
            emb_size = config.dkt_emb_size,
            hidden_size = config.dkt_hidden_size
        ).to(device)
    elif config.model_name == "dkt_plus":
        model = DKT_plus(
            num_q = num_q,
            emb_size = config.dkt_plus_emb_size,
            hidden_size = config.dkt_plus_hidden_size,
        ).to(device)
    #-> 추가적인 모델 정의
    else:
        print("Wrong model_name was used...")

    return model