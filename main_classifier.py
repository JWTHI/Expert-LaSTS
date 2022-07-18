'''
----------------------------------------------------------------------
 BSD 3-Clause License

 Copyright (c) 2022, Jonas Wurst
 All rights reserved.
----------------------------------------------------------------------

'''
import warnings

warnings.filterwarnings("ignore", message='Tensorflow not installed; ParametricUMAP', category=UserWarning)
warnings.filterwarnings("ignore", message='WARNING: spectral ini', category=UserWarning)

import json
import os
import os.path
from time import time
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import umap

from src.datasets_class import triplet_dataloader
from src.scenenet_classifier import make_scenenet
from src.training_class import  train_epoch


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
scene_dir = BASE_DIR+'\\data\\'
MODEL_DIR = BASE_DIR+'\\models\\'
RESULTS_DIR = BASE_DIR+'\\results\\'

NUM_WORKERS = 0

batch_size = 64
epochs = 100
lr = 5*1e-5
betas = (0.5, 0.999)
clip = 100

margin_nn = 1.0
margin_pn = 1.0
margin_pp = 1.0
l2 = 0.01

alpha_recon_meta = {'start':10.0,'end':10.0,'epoch':0,'epoch_delta':0}
alpha_metric_meta = {'start':1.0,'end':1.0,'epoch':0,'epoch_delta':0}
beta_nn_meta = {'start':1.0,'end':1.0,'epoch':0,'epoch_delta':0}
beta_pn_meta = {'start':1.0,'end':1.0,'epoch':0,'epoch_delta':0}
beta_pp_meta = {'start':1.0,'end':1.0,'epoch':0,'epoch_delta':0}
beta_traj_meta = {'start':5.0,'end':5.0,'epoch':0,'epoch_delta':0}
beta_infra_meta = {'start':5.0,'end':5.0,'epoch':0,'epoch_delta':0}
beta_back_traj_meta = {'start':20.0,'end':20.0,'epoch':0,'epoch_delta':0}
beta_back_infra_meta = {'start':10.0,'end':10.0,'epoch':0,'epoch_delta':0}

image_size_px = 100
image_size_mt = 200
orientation = 'ego'
nn_sampling = 'random'
hardest_sampling = False
input_augmentation = False
output_type = 'input'

z_dim_t = 16
z_dim_i = 64
z_dim_m = 64

channels = 1
traj_size = 3

#encoderI_type = "ViT"
encoderI_type = "ResNet-18"
#encoderI_args = {'patch_size':10,'dim':256,'depth':16,'heads':16,'mlp_dim':128}
encoderI_args = None
encoderT_type = "Transformer-Encoder"
#encoderT_type = "LSTM"
encoderT_args = None
merge_type = "FC"
merge_args = {'n_layers':4}

generate_latent_vis = True
plot_loss_curve = True
plot_recon = True
print_every = 1000
save_model = True
load_model = False

#----------------------------------------------------------
# Don't change 
scenenet_args = {'encoderI_type':encoderI_type,'encoderT_type':encoderT_type,'merge_type':merge_type,'encoderI_args':encoderI_args,'encoderT_args':encoderT_args,'merge_args':merge_args,'z_dim_t': z_dim_t,'z_dim_i':z_dim_i,'z_dim_m':z_dim_m,'image_size':image_size_px,'channels':channels,'traj_size':traj_size}
data_args = {'scene_dir':scene_dir,'orientation':orientation,'image_size_px':image_size_px,'image_size_mt':image_size_mt,'batch_size':batch_size,'nn_sampling':nn_sampling,'input_augmentation':input_augmentation,'output_type':output_type}
opitmizer_args = {'lr':lr, 'betas':betas}
scheduler_args = {'alpha_recon_meta':alpha_recon_meta,'alpha_metric_meta':alpha_metric_meta,'beta_nn_meta':beta_nn_meta,'beta_pn_meta':beta_pn_meta,'beta_pp_meta':beta_pp_meta,'beta_traj_meta':beta_traj_meta, 'beta_infra_meta':beta_infra_meta, 'beta_back_traj_meta':beta_back_traj_meta,'beta_back_infra_meta':beta_back_infra_meta}
margin_args = {'margin_nn':margin_nn,'margin_pn':margin_pn,'margin_pp':margin_pp, 'l2':l2}
training_args = {'print_every':print_every,'clip':clip, 'plot_recon':plot_recon,'margin_args':margin_args,'scheduler_args':scheduler_args,'hardest_sampling':hardest_sampling}
general_args ={'epochs':epochs,'generate_latent_vis':generate_latent_vis,'plot_loss_curve':plot_loss_curve,'save_model':save_model,'load_model':load_model}
#
#-----------------------------------------------------------
if hardest_sampling:
    temp_str = 'hard_'
else:
    temp_str = 'rand_'
# Build the name
MODEL_NAME = 'classifier_'+temp_str+str(data_args['nn_sampling'])+'-'+str(data_args['output_type'])+'-'+str(data_args['image_size_mt'])+'-'+str(data_args['image_size_px'])+'-'+str(data_args['orientation'])+'-'+str(data_args['batch_size'])+'_'

if scenenet_args['encoderI_type']=='ViT':
    MODEL_NAME = MODEL_NAME + scenenet_args['encoderI_type']+str(scenenet_args['z_dim_i'])+'-'+str(scenenet_args['encoderI_args']['depth'])+'-'+str(scenenet_args['encoderI_args']['heads'])
elif scenenet_args['encoderI_type']=='ResNet-18':
    MODEL_NAME = MODEL_NAME + scenenet_args['encoderI_type']+str(scenenet_args['z_dim_i'])
else:
    MODEL_NAME = MODEL_NAME + scenenet_args['encoderI_type']+str(scenenet_args['z_dim_i'])

if scenenet_args['encoderT_type']=='LSTM':
    MODEL_NAME = MODEL_NAME+'_'+scenenet_args['encoderT_type']+str(scenenet_args['z_dim_t'])
elif scenenet_args['encoderT_type']=='Transformer-Encoder':
    MODEL_NAME = MODEL_NAME+'_'+scenenet_args['encoderT_type']+str(scenenet_args['z_dim_t'])

if scenenet_args['merge_type']=='FC':
    MODEL_NAME = MODEL_NAME+'_'+scenenet_args['merge_type']+str(scenenet_args['z_dim_m'])+'-'+str(merge_args['n_layers'])
else:
    MODEL_NAME = MODEL_NAME+'_'+scenenet_args['merge_type']+str(scenenet_args['z_dim_m'])

MODEL_NAME = MODEL_NAME+ '_'+str(general_args['epochs']) +'_'+str(int(scheduler_args['alpha_recon_meta']['start']))+'_'+str(int(scheduler_args['alpha_metric_meta']['start']))

model_idx = 1
MODEL_NAME_new = MODEL_NAME
while True:
    if os.path.exists(MODEL_DIR+MODEL_NAME_new+'.ckpt'):
        MODEL_NAME_new = MODEL_NAME+'__'+str(model_idx)
        model_idx = model_idx + 1
    else:
        break

MODEL_NAME = MODEL_NAME_new
RESULTS_DIR = RESULTS_DIR+MODEL_NAME+'\\'

print(MODEL_NAME)
def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 1 * len(loader)
    base_lr = args.learning_rate * args.batch_size / 256
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

def main():
    # Environment stuff
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = torch.cuda.is_available()
    print(cuda)

    # Set up dataloader
    dataloader = triplet_dataloader(**data_args)
    print('Dataloader set up complete.')

    # Set up scenenet
    SceneNet = make_scenenet(**scenenet_args)
    if cuda: SceneNet.cuda()
    optimizer = optim.Adam(SceneNet.parameters(),**opitmizer_args)
    print('SceneNet set up complete.')
    
    
    if general_args['generate_latent_vis'] or general_args['load_model']:
        data_args_2 = data_args
        data_args_2['shuffle'] = False
        dataloader2 = triplet_dataloader(**data_args_2)
        print('Dataloader set up complete.')
        # Load images for latent representation generation
        x_image = np.zeros([len(dataloader2.dataset),1,data_args_2['image_size_px'],data_args_2['image_size_px']])
        x_trajectory = np.zeros([len(dataloader2.dataset),30,3])
        labels = np.zeros([len(dataloader2.dataset),])
        idx_focus = 0
        for idx, tuplets in enumerate(dataloader2):
            x_temp = tuplets['anchor_image']
            x_image[idx_focus:idx_focus+x_temp.shape[0],:,:,:] = x_temp
            x_trajectory[idx_focus:idx_focus+x_temp.shape[0],:,:] = tuplets['anchor_trajectory']
            labels[idx_focus:idx_focus+x_temp.shape[0]] = tuplets['anchor_group']
            idx_focus += x_temp.shape[0]

        x_image = torch.from_numpy(x_image).float()
        x_trajectory = torch.from_numpy(x_trajectory).float()
        z_all = np.zeros([x_image.shape[0],scenenet_args['z_dim_m']])
        reducer = umap.UMAP(n_neighbors=50)#n_epochs=15
        del dataloader2
        
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)

    loss_epoch = []
    epoch_vector = []
    t0 = time()
    results_fn = os.path.join(MODEL_DIR,MODEL_NAME + '.txt')
    #---------------------------------------------------------
    # TRAINING
    #---------------------------------------------------------
    with open(results_fn, 'w') as file:
        print('Begin training.................')
        if general_args['plot_loss_curve']:
            fig_epoch = plt.figure(1)
        for epoch in range(0, epochs):
            # Train the Epoch
            avg_loss = train_epoch(
                SceneNet, cuda, dataloader, optimizer, epoch+1, RESULTS_DIR ,t0=t0,epochs=general_args['epochs'], lr=opitmizer_args['lr'], **training_args)
            
            if general_args['plot_loss_curve']:
                # Visualize the loss
                loss_epoch.append(avg_loss.cpu().numpy())
                epoch_vector.append(epoch)
                plt.figure(1)
                plt.cla()
                plt.plot(epoch_vector,loss_epoch,color='k')
                plt.pause(0.05)
                plt.figure(1)
                plt.savefig(RESULTS_DIR+MODEL_NAME + '.pdf')
            
            if general_args['generate_latent_vis']:
                    # Visualize the latent Space
                SceneNet.eval()
                
                for i in range(0,x_image.shape[0],data_args['batch_size']):
                    image_batch = x_image[i:min(x_image.shape[0],i+data_args['batch_size']),:,:,:].cuda()
                    trajectory_batch = x_trajectory[i:min(x_trajectory.shape[0],i+data_args['batch_size']),:,:].cuda()
                    z = SceneNet.encode(image_batch,trajectory_batch)
                    z = (Variable(z).data).cpu().numpy()
                    z_all[i:min(x_image.shape[0],i+data_args['batch_size']),:] = z

                embedding = reducer.fit_transform(z_all)
                plt.figure(2)
                plt.cla()
                plt.scatter(embedding[:, 0],embedding[:, 1],c=labels, cmap='viridis', s=5)
                plt.savefig(RESULTS_DIR+'emb'+str(epoch)+'.pdf')
                plt.close()

                SceneNet.train()

    # Save model after last epoch
    if general_args['save_model']:
        model_fn = os.path.join(MODEL_DIR, MODEL_NAME + '.ckpt')
        torch.save(SceneNet.state_dict(), model_fn)
        all_args = {'scenenet_args':scenenet_args, 'data_args':data_args, 'opitmizer_args':opitmizer_args, 'scheduler_args':scheduler_args ,'margin_args':margin_args ,'training_args':training_args ,'general_args':general_args}
        json_fn = os.path.join(MODEL_DIR, MODEL_NAME + '.json')
        with open(json_fn, 'w') as fp:
            json.dump(all_args, fp)
    
    SceneNet.eval()
                
    # for i in range(0,x_image.shape[0],data_args['batch_size']):
    #     image_batch = x_image[i:min(x_image.shape[0],i+data_args['batch_size']),:,:,:].cuda()
    #     trajectory_batch = x_trajectory[i:min(x_trajectory.shape[0],i+data_args['batch_size']),:,:].cuda()
    #     z = SceneNet.encode(image_batch,trajectory_batch)
    #     z = (Variable(z).data).cpu().numpy()
    #     z_all[i:min(x_image.shape[0],i+data_args['batch_size']),:] = z
    
    # embedding_fn = RESULTS_DIR + 'lat_embedding.json'
    # with open(embedding_fn, 'w') as outfile:
    #     json.dump(z_all.tolist(), outfile)

    

if __name__ == '__main__':
    main()
