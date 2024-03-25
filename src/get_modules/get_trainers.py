from trainers.dkt_trainer import DKT_trainer
from trainers.dkvmn_trainer import DKVMN_trainer
from trainers.sakt_trainer import SAKT_trainer
from trainers.akt_trainer import AKT_trainer
from trainers.gkt_trainer import GKT_trainer

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
    elif config.model_name == "gkt_pam" or config.model_name == "gkt_mha":
        trainer = GKT_trainer(
            model = model,
            optimizer = optimizer,
            n_epochs = config.n_epochs,
            device = device,
            num_q = num_q,
            crit = crit,
            max_seq_len=config.max_seq_len      
        )

    return trainer
