from models.dkt import DKT
from models.dkt_plus import DKT_plus
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.akt import AKT
from models.dkt_c_q import DKT_c_q
from models.dkt_c_rasch import DKT_c_rasch
from models.dkt_c_q_ctt import DKT_c_q_ctt
from models.dkvmn_c_q import DKVMN_c_q
from models.dkvmn_c_rasch import DKVMN_c_rasch
from models.dkvmn_c_q_ctt import DKVMN_c_q_ctt

def get_models(num_q, num_r, num_pid, num_diff, device, config):

    if config.model_name == "dkt":
        model = DKT(
            num_q = num_q,
            emb_size = config.dkt_emb_size,
            hidden_size = config.dkt_hidden_size
        ).to(device)
    elif config.model_name == "dkt_c_q":
        model = DKT_c_q(
            num_q = num_q,
            num_r = num_r,
            num_pid = num_pid,
            emb_size = config.dkt_emb_size,
            hidden_size = config.dkt_hidden_size
        ).to(device)
    elif config.model_name == "dkt_c_rasch":
        model = DKT_c_rasch(
            num_q = num_q,
            num_r = num_r,
            num_pid = num_pid,
            emb_size = config.dkt_emb_size,
            hidden_size = config.dkt_hidden_size
        ).to(device)
    elif config.model_name == "dkt_c_q_ctt":
        model = DKT_c_q_ctt(
            num_q = num_q,
            num_r = num_r,
            num_pid = num_pid,
            num_diff = num_diff,
            emb_size = config.dkt_emb_size,
            hidden_size = config.dkt_hidden_size
        ).to(device)
    elif config.model_name == "dkt_plus":
        model = DKT_plus(
            num_q = num_q,
            emb_size = config.dkt_plus_emb_size,
            hidden_size = config.dkt_plus_hidden_size,
        ).to(device)
    elif config.model_name == "dkvmn":
        model = DKVMN(
            num_q = num_q,
            dim_s = config.dkvmn_dim_s, #default = 50
            size_m = config.dkvmn_size_m #default = 20
        ).to(device)
    elif config.model_name == "dkvmn_c_q":
        model = DKVMN_c_q(
            num_q = num_q,
            num_pid = num_pid,
            dim_s = config.dkvmn_dim_s, #default = 50
            size_m = config.dkvmn_size_m #default = 20
        ).to(device)
    elif config.model_name == "dkvmn_c_rasch":
        model = DKVMN_c_rasch(
            num_q = num_q,
            num_pid = num_pid,
            dim_s = config.dkvmn_dim_s, #default = 50
            size_m = config.dkvmn_size_m #default = 20
        ).to(device)
    elif config.model_name == "dkvmn_c_q_ctt":
        model = DKVMN_c_q_ctt(
            num_q = num_q,
            num_pid = num_pid,
            num_diff = num_diff,
            dim_s = config.dkvmn_dim_s, #default = 50
            size_m = config.dkvmn_size_m #default = 20
        ).to(device)
    elif config.model_name == "sakt":
        model = SAKT(
            num_q = num_q,
            n = config.sakt_n, #default = 100
            d = config.sakt_d, #default = 100
            num_attn_heads = config.sakt_num_attn_heads, #default = 5
            device = device #어쩔 수 없이 추가 ㅜㅜ
        ).to(device)
    elif config.model_name == "akt":
        model = AKT(
            n_question=num_q,
            n_pid=num_pid,
            d_model=config.akt_d_model,
            n_blocks=config.akt_n_block,
            kq_same=config.akt_kq_same,
            dropout=config.akt_dropout_p,
            model_type="akt"
        ).to(device)
    else:
        print("Wrong model_name was used...")

    return model