from paddle import unsqueeze
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import one_hot

def personal_pred_visualizer(model, model_path, test_loader, device, img_name, num_q):

    img_path = './imgs/' + img_name + '_persoanl_pred'

    model = model
    checkpoint = torch.load(model_path)
    model.load_state_dict( checkpoint['model'] )
    model.to(device)

    #test
    with torch.no_grad():
        for data in test_loader:
            
            q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs = data

            random_idx = random.randint(0, len(q_seqs) - 1) #|len(q_seqs)| = bs

            q_seqs = q_seqs[random_idx].to(device)
            r_seqs = r_seqs[random_idx].to(device)
            qshft_seqs = qshft_seqs[random_idx].to(device)
            rshft_seqs = rshft_seqs[random_idx].to(device)
            mask_seqs = mask_seqs[random_idx].to(device)

            masked_q_seqs = torch.masked_select( q_seqs, mask_seqs )
            masked_r_seqs = torch.masked_select( r_seqs, mask_seqs )

            masked_unsqueeze_q_seqs = torch.unsqueeze(masked_q_seqs, 0)
            masked_unsqueeze_r_seqs = torch.unsqueeze(masked_r_seqs, 0)

            y_hat = model( masked_unsqueeze_q_seqs.long(), masked_unsqueeze_r_seqs.long() )

            pred = y_hat[0, :, :].detach().cpu().numpy()
        
            plt.subplot(121)
            plt.figure(figsize=(12,5))
            plt.imshow(pred)
            plt.xlabel('Index of item')
            plt.ylabel('Number of responses')
            plt.colorbar()

            plt.savefig(img_path + '_imshow' + '.jpg')

            plt.subplot(122)
            plt.figure(figsize=(12,5))
            plt.plot(np.mean(pred ,axis=1), c='red')
            plt.plot(pred, c='black', alpha=0.15)
            plt.legend(['Mean', 'Each item'])
            plt.xlabel('Number of responses')
            
            plt.savefig(img_path + '_plot' + '.jpg')

            #반복못하도록 한번만 돌리기
            break