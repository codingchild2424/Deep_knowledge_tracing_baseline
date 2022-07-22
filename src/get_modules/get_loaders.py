from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from utils import collate_fn, pid_collate_fn, dkt_pid_collate_fn, dkt_pid_diff_collate_fn, pid_diff_collate_fn
from dataloaders.assist2015_loader import ASSIST2015
from dataloaders.assist2009_loader import ASSIST2009
from dataloaders.algebra2005_loader import ALGEBRA2005
from dataloaders.algebra2006_loader import ALGEBRA2006
from dataloaders.slepemapy_loader import SLEPEMAPY
from dataloaders.ednet_loader import EDNET
from dataloaders.assist2017_loader import ASSIST2017
from dataloaders.statics_loader import STATICS
from dataloaders.assist2009_pid_loader import ASSIST2009_PID
from dataloaders.assist2017_pid_loader import ASSIST2017_PID
from dataloaders.assist2012_loader import ASSIST2012
from dataloaders.assist2012_pid_loader import ASSIST2012_PID
from dataloaders.algebra2005_pid_loader import ALGEBRA2005_PID
from dataloaders.algebra2006_pid_loader import ALGEBRA2006_PID
<<<<<<< HEAD
from dataloaders.slepemapy_loader import SLEPEMAPY
from dataloaders.slepemapy_pid_loader import SLEPEMAPY_PID
=======
from dataloaders.assist2009_pid_diff_loader import ASSIST2009_PID_DIFF
>>>>>>> 3a965899191a4ca2ccac25dc3cfd8ddec5de0d7b

#get_loaders를 따로 만들고, 이 함수를 train에서 불러내기
def get_loaders(config, idx=None):

    #1. dataset 선택
    if config.dataset_name == "assist2015":
        dataset = ASSIST2015(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "assist2009":
        dataset = ASSIST2009(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "assist2012":
        dataset = ASSIST2012(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "algebra2005":
        dataset = ALGEBRA2005(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "algebra2006":
        dataset = ALGEBRA2006(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "slepemapy":
        dataset = SLEPEMAPY(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "ednet":
        dataset = EDNET(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "assist2017":
        dataset = ASSIST2017(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "statics":
        dataset = STATICS(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = None
        num_diff = None
        collate = collate_fn
    elif config.dataset_name == "assist2009_pid":
        dataset = ASSIST2009_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "dkt_assist2009_pid":
        dataset = ASSIST2009_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = dkt_pid_collate_fn
    elif config.dataset_name == "assist2017_pid":
        dataset = ASSIST2017_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "assist2012_pid":
        dataset = ASSIST2012_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "algebra2005_pid":
        dataset = ALGEBRA2005_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "algebra2006_pid":
        dataset = ALGEBRA2006_PID(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = None
        collate = pid_collate_fn
<<<<<<< HEAD
    elif config.dataset_name == "slepemapy_pid":
        dataset = SLEPEMAPY_PID(config.max_seq_len)
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        collate = pid_collate_fn
=======
    elif config.dataset_name == "dkt_assist2009_pid_diff":
        dataset = ASSIST2009_PID_DIFF(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = dataset.num_diff
        collate = dkt_pid_diff_collate_fn
    elif config.dataset_name == "assist2009_pid_diff":
        dataset = ASSIST2009_PID_DIFF(config.max_seq_len)  
        num_q = dataset.num_q
        num_r = dataset.num_r
        num_pid = dataset.num_pid
        num_diff = dataset.num_diff
        collate = pid_diff_collate_fn
>>>>>>> 3a965899191a4ca2ccac25dc3cfd8ddec5de0d7b
    else:
        print("Wrong dataset_name was used...")

    if config.fivefold == True:

        first_chunk = Subset(dataset, range( int(len(dataset) * 0.2) ))
        second_chunk = Subset(dataset, range( int(len(dataset) * 0.2), int(len(dataset)* 0.4) ))
        third_chunk = Subset(dataset, range( int(len(dataset) * 0.4), int(len(dataset) * 0.6) ))
        fourth_chunk = Subset(dataset, range( int(len(dataset) * 0.6), int(len(dataset) * 0.8) ))
        fifth_chunk = Subset(dataset, range( int(len(dataset) * 0.8), int(len(dataset)) ))

        #idx는 함수에서 매개변수로 받아옴
        if idx == 0:
            #전체의 0.8
            train_dataset = ConcatDataset([second_chunk, third_chunk, fourth_chunk, fifth_chunk])

            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            #전체의 0.2
            test_dataset = first_chunk
        elif idx == 1:
            train_dataset = ConcatDataset([first_chunk, third_chunk, fourth_chunk, fifth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = second_chunk
        elif idx == 2:
            train_dataset = ConcatDataset([first_chunk, second_chunk, fourth_chunk, fifth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = third_chunk
        elif idx == 3:
            train_dataset = ConcatDataset([first_chunk, second_chunk, third_chunk, fifth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = fourth_chunk
        elif idx == 4:
            train_dataset = ConcatDataset([first_chunk, second_chunk, third_chunk, fourth_chunk])
            valid_size = int( len(train_dataset) * config.valid_ratio) #train의 0.1
            train_size = int( len(train_dataset) ) - valid_size #train의 0.9
            
            train_dataset, valid_dataset = random_split(
                train_dataset, [ train_size, valid_size ]
            )
            test_dataset = fifth_chunk
    else:
        #3. train, test 사이즈 나누기
        train_size = int( len(dataset) * config.train_ratio * (1 - config.valid_ratio)) #0.8의 0.9
        valid_size = int( len(dataset) * config.train_ratio * config.valid_ratio) #0.8의 0.1
        test_size = len(dataset) - (train_size + valid_size) #전체의 0.2

        #train, test 사이즈 나누기
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, [ train_size, valid_size, test_size ]
            )

    #4. DataLoader 불러오기
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        shuffle = True,
        collate_fn = collate
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size = config.batch_size,
        shuffle = False,
        collate_fn = collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size,
        shuffle = False, #test
        collate_fn = collate
    )

    return train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff