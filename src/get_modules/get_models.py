from models.dkt import DKT
from models.dkvmn import DKVMN
from models.sakt import SAKT
from models.akt import AKT
from models.gkt import PAM, MHA

from models.gkt import PAM, MHA

def get_models(num_q, num_r, num_pid, num_diff, device, config):

    if config.model_name == "dkt":
        model = DKT(
            num_q = num_q,
            emb_size = config.dkt_emb_size,
            hidden_size = config.dkt_hidden_size
        ).to(device)
    elif config.model_name == "dkvmn":
        model = DKVMN(
            num_q = num_q,
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
    # GKT
    elif config.model_name == "gkt_pam":
        model = PAM(
            num_q=num_q,
            hidden_size=config.gkt_hidden_size,
            num_attn_heads=config.gkt_num_attn_heads
        ).to(device)
    elif config.model_name == "gkt_mha":
        model = MHA(
            num_q=num_q,
            hidden_size=config.gkt_hidden_size,
            num_attn_heads=config.gkt_num_attn_heads
        ).to(device)

    else:
        print("Wrong model_name was used...")

    return model