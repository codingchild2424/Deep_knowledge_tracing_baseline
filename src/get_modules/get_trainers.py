from trainers.dkt_trainer import DKT_trainer
from trainers.dkt_plus_trainer import DKT_plus_trainer
from trainers.dkvmn_trainer import DKVMN_trainer
from trainers.sakt_trainer import SAKT_trainer
from trainers.akt_trainer import AKT_trainer
from trainers.dkt_c_q_trainer import DKT_c_q_trainer
from trainers.dkt_c_rasch_trainer import DKT_c_rasch_trainer
from trainers.dkt_c_q_ctt_trainer import DKT_c_q_ctt_trainer
from trainers.dkvmn_c_q_trainer import DKVMN_c_q_trainer
from trainers.dkvmn_c_rasch_trainer import DKVMN_c_rasch_trainer
from trainers.dkvmn_c_q_ctt_trainer import DKVMN_c_q_ctt_trainer
from trainers.sakt_c_q_trainer import SAKT_c_q_trainer
from trainers.sakt_c_rasch_trainer import SAKT_c_rasch_trainer
from trainers.sakt_c_q_ctt_trainer import SAKT_c_q_ctt_trainer

def get_trainers(model, optimizer, device, num_q, crit, config):

    #trainer 실행
    if config.model_name == "dkt":
        trainer = DKT_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len
        )
    elif config.model_name == "dkt_c_q":
        trainer = DKT_c_q_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len
        )
    elif config.model_name == "dkt_c_rasch":
        trainer = DKT_c_rasch_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len
        )
    elif config.model_name == "dkt_c_q_ctt":
        trainer = DKT_c_q_ctt_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len
        )
    elif config.model_name == "dkt_plus":
        trainer = DKT_plus_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len,
            lambda_r = config.dkt_plus_lambda_r,
            lambda_w1 = config.dkt_plus_lambda_w1,
            lambda_w2 = config.dkt_plus_lambda_w2
        )
    elif config.model_name == "dkvmn":
        trainer = DKVMN_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len
        )
    elif config.model_name == "dkvmn_c_q":
        trainer = DKVMN_c_q_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len
        )
    elif config.model_name == "dkvmn_c_rasch":
        trainer = DKVMN_c_rasch_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len
        )
    elif config.model_name == "dkvmn_c_q_ctt":
        trainer = DKVMN_c_q_ctt_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len
        )
    elif config.model_name == "sakt":
        trainer = SAKT_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len           
        )
    elif config.model_name == "sakt_c_q":
        trainer = SAKT_c_q_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len           
        )
    elif config.model_name == "sakt_c_rasch":
        trainer = SAKT_c_rasch_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len           
        )
    elif config.model_name == "sakt_c_q_ctt":
        trainer = SAKT_c_q_ctt_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len           
        )
    elif config.model_name == "akt":
        trainer = AKT_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len           
        )

    return trainer
