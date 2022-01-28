import torch
from copy import deepcopy

from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics

from tqdm import tqdm

class Trainer():

    def __init__(self, model, optimizer, num_epochs, device, num_q):
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.num_q = num_q
    
    def _train(self, train_loader):

        auc_score = 0
        y_trues, y_scores = [], []

        for data in tqdm(train_loader):
            self.model.train()
            q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
            q_seqs = q_seqs.to(self.device) #|q_seqs| = (bs, sq) -> [[58., 58., 58., -0., -0., -0., -0., ...], [58., 58., 58., -0., -0., -0., -0., ...]...]
            r_seqs = r_seqs.to(self.device) #|r_seqs| = (bs, sq) -> [[1., 1., 0., -0., -0., -0., -0., ...], [1., 1., 0., -0., -0., -0., -0., ...]...]
            qshft_seqs = qshft_seqs.to(self.device) #|qshft_seqs| = (bs, sq) -> [[58., 58., 58., -0., -0., -0., -0., ...], [58., 58., 58., -0., -0., -0., -0., ...]...]
            rshft_seqs = rshft_seqs.to(self.device) #|rshft_seqs| = (bs, sq) -> [[1., 1., 0., -0., -0., -0., -0., ...], [1., 1., 0., -0., -0., -0., -0., ...]...]
            mask_seqs = mask_seqs.to(self.device) #|mask_seqs| = (bs, sq) -> [[True,  True,  True,  ..., False, False, False], [True,  True,  True,  ..., False, False, False]..]

            y_hat = self.model( q_seqs.long(), r_seqs.long() ) #|y_hat| = (bs, sq, self.num_q) -> tensor([[[0.6938, 0.7605, ..., 0.7821], [0.8366, 0.6598,  ..., 0.8514],..)

            #|qshft_seqs| = (bs, sq) -> tensor([[43., 43., 79.,  ..., -0., -0., -0.], [59., 15., 47.,  ..., -0., -0., -0.],...])
            #|self.num_q| = 100
            
            one_hot_vectors = one_hot(qshft_seqs.long(), self.num_q) #|one_hot_vectors| = (bs, sq, self.num_q) -> tensor([[[0, 0, 0,  ..., 0, 0, 0], [0, 0, 0,  ..., 0, 0, 0], [0, 0, 0,  ..., 0, 0, 0],..], [[]])

            y_hat = ( y_hat * one_hot_vectors ).sum(-1) #|y_hat| = (bs, sq) -> tensor([[0.5711, 0.7497, 0.8459,  ..., 0.6606, 0.6639, 0.6702], [0.5721, 0.6495, 0.6956,  ..., 0.6677, 0.6687, 0.6629],

            y_hat = torch.masked_select(y_hat, mask_seqs) #|y_hat| = () -> tensor([0.7782, 0.8887, 0.7638,  ..., 0.8772, 0.8706, 0.8831])

            correct = torch.masked_select(rshft_seqs, mask_seqs) #|correct| = () -> tensor([0., 1., 1.,  ..., 1., 1., 1.])

            self.optimizer.zero_grad()
            loss = binary_cross_entropy(y_hat, correct) #|loss| = () -> ex) 0.5432

            loss.backward()
            self.optimizer.step()

            y_trues.append(correct)
            y_scores.append(y_hat)

        y_trues = torch.cat(y_trues).detach().cpu().numpy() #|y_tures| = () -> [0. 0. 0. ... 1. 1. 1.]
        y_scores = torch.cat(y_scores).detach().cpu().numpy() #|y_scores| = () ->  tensor(0.5552)

        auc_score += metrics.roc_auc_score( y_trues, y_scores ) #|metrics.roc_auc_score( y_trues, y_scores )| = () -> 0.6203433289463159

        return auc_score

    def _test(self, test_loader):

        auc_score = 0
        y_trues, y_scores = [], []

        with torch.no_grad():
            for data in tqdm(test_loader):
                self.model.eval()
                q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data #collate에 정의된 데이터가 나옴
                q_seqs = q_seqs.to(self.device)
                r_seqs = r_seqs.to(self.device)
                qshft_seqs = qshft_seqs.to(self.device)
                rshft_seqs = rshft_seqs.to(self.device)
                mask_seqs = mask_seqs.to(self.device)

                y_hat = self.model( q_seqs.long(), r_seqs.long() )
                y_hat = (y_hat * one_hot(qshft_seqs.long(), self.num_q)).sum(-1)

                y_hat = torch.masked_select(y_hat, mask_seqs)
                correct = torch.masked_select(rshft_seqs, mask_seqs)

                y_trues.append(correct)
                y_scores.append(y_hat)

        y_trues = torch.cat(y_trues).detach().cpu().numpy()
        y_scores = torch.cat(y_scores).detach().cpu().numpy()

        auc_score += metrics.roc_auc_score( y_trues, y_scores )

        return auc_score

    def train(self, train_loader, test_loader):
        
        highest_auc_score = 0
        best_model = None

        for epoch_index in range(self.num_epochs):
            
            print("Epoch(%d/%d) start" % (
                epoch_index + 1,
                self.num_epochs
            ))

            train_auc_score = self._train(train_loader)
            test_auc_score = self._test(test_loader)

            if test_auc_score >= highest_auc_score:
                highest_auc_score = test_auc_score
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d) result: train_auc_score=%.4f  test_auc_score=%.4f  highest_auc_score=%.4f" % (
                epoch_index + 1,
                self.num_epochs,
                train_auc_score,
                test_auc_score,
                highest_auc_score,
            ))

        print("\n")
        print("The Highest_Auc_Score in Training Session is %.4f" % (
                highest_auc_score,
            ))
        print("\n")
        
        # 가장 최고의 모델 복구    
        self.model.load_state_dict(best_model)