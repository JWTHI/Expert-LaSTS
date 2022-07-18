'''
    
----------------------------------------------------------------------
 BSD 3-Clause License

 Copyright (c) 2022, Jonas Wurst
 All rights reserved.
----------------------------------------------------------------------

'''
from time import time

import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
#import umap
def scheduler_minor(epoch,meta):
    if epoch <meta['epoch']:
        value = meta['start']
    elif epoch >=meta['epoch']+meta['epoch_delta']:
        value = meta['end']
    else:
        # TODO not valid yet
        value_delta = (meta['end']-meta['start'])/meta['epoch_delta']
        value = meta['start']+value_delta*(epoch-meta['epoch'])
    return value

def scheduler(epoch=None,alpha_recon_meta=None,alpha_metric_meta=None,beta_nn_meta=None,beta_pn_meta=None,beta_pp_meta=None,beta_traj_meta=None,beta_infra_meta=None,beta_back_traj_meta=None,beta_back_infra_meta=None):
    loss_weight_args = {'beta_pp':scheduler_minor(epoch,beta_pp_meta),
        'beta_pn':scheduler_minor(epoch,beta_pn_meta),
        'beta_nn':scheduler_minor(epoch,beta_nn_meta),
        'alpha_recon':scheduler_minor(epoch,alpha_recon_meta),
        'alpha_metric':scheduler_minor(epoch,alpha_metric_meta),
        'beta_traj':scheduler_minor(epoch,beta_traj_meta),
        'beta_infra':scheduler_minor(epoch,beta_infra_meta),
        'beta_back_infra':scheduler_minor(epoch,beta_back_infra_meta),
        'beta_back_traj':scheduler_minor(epoch,beta_back_traj_meta)}
    return loss_weight_args

def adjust_learning_rate(epochs,learning_rate,batch_size, optimizer, loader, step):
    max_steps = epochs * len(loader)
    warmup_steps = 1 * len(loader)
    base_lr = learning_rate * batch_size / 64
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def prep_tuplets(tuplets, cuda):
    """
    Takes a batch of tuplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    tuplets_out = dict((k, Variable(v)) for k,v in tuplets.items())
    tuplets = tuplets_out
    
    if cuda:
        tuplets_out = dict((k, v.cuda()) for k,v in tuplets.items())
    return tuplets_out



def train_epoch(model, cuda, dataloader, optimizer, epoch, RESULTS_DIR, print_every=100, t0=None,clip=100,lr=1e-3,epochs=100,plot_recon=False,scheduler_args=None,margin_args=None,hardest_sampling=True):
    """
    Trains a model for one epoch using the provided dataloader.
    """
    model.train()

    if t0 is None:
        t0 = time()
    sum_loss = 0
    n_train, n_batches = len(dataloader.dataset), len(dataloader)

    #loss_weight_args = scheduler(epoch=epoch,**scheduler_args)
    #step = (epoch-1)*n_train
    
    #------------------------------------------
    # Training
    #------------------------------------------
    for idx, tuplets in enumerate(dataloader):
        # Optimize for a batch
        #tuplets = dataloader.dataset.get_negatives_in_batch(tuplets)
        tuplets_out = prep_tuplets(tuplets, cuda)
        #step += idx
        #lr = adjust_learning_rate(epochs,lr,dataloader.batch_size,optimizer, dataloader, step)
        optimizer.zero_grad()
        loss = model.loss(tuplets_out)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        # Gather and Print results
        sum_loss += loss.data
        #sum_loss_metric += loss_metric.data
        #sum_loss_recon += reconstruction_loss.data
        if (idx + 1) * dataloader.batch_size % print_every == 0:
            print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
                epoch, (idx + 1) * dataloader.batch_size, n_train,
                100 * (idx + 1) / n_batches, sum_loss/(idx + 1)))
        
        #
    avg_loss = sum_loss / n_batches
    #avg_l_m = sum_loss_metric / n_batches
    #avg_l_r = sum_loss_recon / n_batches
    print('Finished epoch {}: {:0.3f}s'.format(epoch, time()-t0))
    print('  Average loss: {:0.4f}'.format(avg_loss))
    #print('  Average l_metric: {:0.4f}'.format(avg_l_m))
    #print('  Average l_recon: {:0.4f}\n'.format(avg_l_r))

    '''
    if plot_recon:
        # Visualize the latent Space
        model.eval()
        # Get z for all images
        
        z = model.encode(tuplets_out['anchor_image'],tuplets_out['anchor_trajectory'])
        x_rand = (Variable(tuplets_out['anchor_merged']).data).cpu().numpy()
        x_pred = (Variable(model.decode(z)).data).cpu().numpy()
        dummy_layer = np.zeros(x_rand[0][0].shape)

        anchor_rand = (Variable(tuplets_out['anchor_image']).data).cpu().numpy()
        pp_rand = (Variable(tuplets_out['pp_image']).data).cpu().numpy()
        pn_rand = (Variable(tuplets_out['pn_image']).data).cpu().numpy()
        nn_rand = (Variable(tuplets_out['nn_image']).data).cpu().numpy()
        
        # Visualize reconstruction
        fig, axes = plt.subplots(4, 10,num=3)
        base_count = 0
        for row_idx in np.arange(0,4,2):
            for idx in np.arange(10):
                if base_count+idx ==x_rand.shape[0]-1:
                    break
                image_merged = np.dstack((x_rand[base_count+idx,0,:,:],dummy_layer,x_rand[base_count+idx,1,:,:]))
                image_merged_2 = np.dstack((x_pred[base_count+idx,0,:,:],dummy_layer,x_pred[base_count+idx,1,:,:]))
                
                axes[row_idx  ,idx].imshow( np.clip(image_merged, 0, 1))
                axes[row_idx  ,idx].axis('off')
                axes[row_idx+1,idx].imshow(np.clip(image_merged_2, 0, 1))
                axes[row_idx+1,idx].axis('off')
            base_count =+10
        plt.savefig(RESULTS_DIR+'recon_'+str(epoch)+'.pdf')
        plt.close()

        
        fig, axes = plt.subplots(4, 10,num=3)
        base_count = 0 
        for idx in np.arange(10):
            if idx ==anchor_rand.shape[0]-1:
                break
                
            axes[0,idx].imshow( np.clip(anchor_rand[idx,0,:,:], 0, 1))
            axes[0,idx].axis('off')
            axes[1,idx].imshow( np.clip(pp_rand[idx,0,:,:], 0, 1))
            axes[1,idx].axis('off')
            axes[2,idx].imshow( np.clip(pn_rand[idx,0,:,:], 0, 1))
            axes[2,idx].axis('off')
            axes[3,idx].imshow( np.clip(nn_rand[idx,0,:,:], 0, 1))
            axes[3,idx].axis('off')
        
        #plt.savefig(RESULTS_DIR+'recon_appnn_'+str(epoch)+'.pdf')
        #plt.close()
        

        #plt.imshow(np.clip(image_merged, 0, 1))
        #plt.savefig(RESULTS_DIR+'recon_'+str(epoch)+'_1.pdf')
        #plt.close()

        #plt.imshow(np.clip(image_merged_2, 0, 1))
        #plt.savefig(RESULTS_DIR+'recon_'+str(epoch)+'_2.pdf')
        #plt.close()
    
        model.train()
    '''
    return avg_loss