from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from utils import collate_fn
from dataloaders.assist2015_loader import ASSIST2015
from dataloaders.assist2009_loader import ASSIST2009
from dataloaders.algebra2005_loader import ALGEBRA2005
from dataloaders.algebra2006_loader import ALGEBRA2006
from dataloaders.slepemapy_loader import SLEPEMAPY
from dataloaders.ednet_loader import EDNET

#get_loaders를 따로 만들고, 이 함수를 train에서 불러내기
def get_loaders(config, idx=None):

    #1. dataset 선택
    if config.dataset_name == "assist2015":
        dataset = ASSIST2015(config.max_seq_len)
    elif config.dataset_name == "assist2009":
        dataset = ASSIST2009(config.max_seq_len)
    elif config.dataset_name == "algebra2005":
        dataset = ALGEBRA2005(config.max_seq_len)
    elif config.dataset_name == "algebra2006":
        dataset = ALGEBRA2006(config.max_seq_len)
    elif config.dataset_name == "slepemapy":
        dataset = SLEPEMAPY(config.max_seq_len)
    elif config.dataset_name == "ednet":
        dataset = EDNET(config.max_seq_len)
    else:
        print("Wrong dataset_name was used...")

    #2. unique한 question의 갯수(num_q) 받아오기
    num_q = dataset.num_q
    num_r = dataset.num_r

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
        collate_fn = collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size = config.batch_size,
        shuffle = False,
        collate_fn = collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size,
        shuffle = False, #test
        collate_fn = collate_fn
    )

    return train_loader, valid_loader, test_loader, num_q, num_r