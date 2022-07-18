'''
This file is based on:
    https://github.com/ermongroup/tile2vec/blob/master/src/tilenet.py
    
----------------------------------------------------------------------
 BSD 3-Clause License

 Copyright (c) 2022, Jonas Wurst
 All rights reserved.
----------------------------------------------------------------------

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from src.vit_pytorch import ViT
from src.encoder_pytorch import EncoderFull

class SceneNet(nn.Module):
    def __init__(self, encoderI_type=None, encoderT_type=None, merge_type=None, encoderI_args=None, encoderT_args=None, merge_args=None,z_dim_t=32,z_dim_i=64,z_dim_m=64,image_size=200,channels=1,traj_size=3):
        super(SceneNet, self).__init__()
        self.encoder = EncoderFull(encoderI_type=encoderI_type,
            encoderT_type=encoderT_type,
            merge_type=merge_type,
            encoderI_args=encoderI_args,
            encoderT_args=encoderT_args,
            merge_args=merge_args,
            z_dim_t=z_dim_t,
            z_dim_i=z_dim_i,
            z_dim_m=z_dim_m,
            image_size=image_size,
            channels=channels,
            traj_size=traj_size
            )
        if image_size == 400:
            self.layerDec1 = nn.ConvTranspose2d(z_dim_m, 32,5,2)
            self.layerDec2 = nn.ConvTranspose2d(32, 64,7,4)
            self.layerDec3 = nn.ConvTranspose2d(64, 64,10,4)
            self.layerDec4 = nn.ConvTranspose2d(64, 2,12,4)
        elif image_size == 200:
            self.layerDec1 = nn.ConvTranspose2d(z_dim_m, 32,6,2)
            self.layerDec2 = nn.ConvTranspose2d(32, 64,10,2)
            self.layerDec3 = nn.ConvTranspose2d(64, 64,10,2)
            self.layerDec4 = nn.ConvTranspose2d(64, 2,12,4)
        elif image_size == 100:
            #self.layerDec1 = nn.ConvTranspose2d(z_dim_m, 32,6,2)
            #self.layerDec2 = nn.ConvTranspose2d(32, 64,8,2)
            #self.layerDec3 = nn.ConvTranspose2d(64, 64,11,2)
            #self.layerDec4 = nn.ConvTranspose2d(64, 2,12,2)
            self.layerDec1 = nn.ConvTranspose2d(z_dim_m, 64,6,2)
            self.layerDec2 = nn.ConvTranspose2d(64, 128,8,2)
            self.layerDec3 = nn.ConvTranspose2d(128, 128,8,2)
            self.layerDec4 = nn.ConvTranspose2d(128, 128,8,2)
            self.layerDec5 = nn.ConvTranspose2d(128, 64,6,1)
            self.layerDec6 = nn.ConvTranspose2d(64, 2,6,1)
        

    def encode(self, x_image,x_traj):
        z = self.encoder.forward(x_image,x_traj)
        return z
    
    def decode(self, z):
        x = z.view((z.size(0),z.size(1),1,1))
        x = self.layerDec1(x)
        x = F.relu(x)
        x = self.layerDec2(x)
        x = F.relu(x)
        x = self.layerDec3(x)
        x = F.relu(x)
        x = self.layerDec4(x)
        x = F.relu(x)
        x = self.layerDec5(x)
        x = F.relu(x)
        x = self.layerDec6(x)
        x = torch.sigmoid(x)
        return x

    def forward(self, x_image,x_traj):
        z = self.encode(x_image,x_traj)
        x = self.decode(z)
        return x 

    def metric_loss(self,z_a=None, z_pp=None, z_pn=None, z_nn=None, d_in_pp=None, pn_valid=None, graphID_a=None, graphID_pp=None, graphID_pn=None, graphID_nn=None, margin_nn=None,margin_pn=None,margin_pp=None, l2=None, beta_pp=None, beta_pn=None, beta_nn=None,negatives_a=None,negatives_pn=None,negatives_pp=None,negatives_nn=None,hardest_sampling=True):
        d_pn = ((z_a - z_pn) ** 2).sum(dim=1)
        d_pp = ((z_a - z_pp) ** 2).sum(dim=1)
        d_pp_margin =  d_pp + F.relu(margin_pp-d_pp)
        z_nn_inside = torch.zeros_like(z_nn)

        nn_correction = []
        for i in range(z_a.shape[0]):
            # Get all possible points and stack them
            z_nn_a = z_a[negatives_a[i,:],:]
            z_nn_pp = z_pp[negatives_pp[i,:],:]
            z_nn_pn = z_pn[negatives_pn[i,:],:]
            z_nn_nn = z_nn[negatives_nn[i,:],:]
            z_all_negatives = torch.cat((z_nn_a,z_nn_pp,z_nn_pn,z_nn_nn))
            d_nn_all = ((z_a[i,:] - z_all_negatives) ** 2).sum(dim=1)

            # Finding Semi-Hards
            mask = torch.gt(d_nn_all, d_pn[i]) & torch.lt( d_nn_all,d_pn[i] + margin_nn) # d_nn> d_pn & d_nn<d_pn+margin
            d_nn_all_sub = d_nn_all[mask]
            z_all_negatives_mask = z_all_negatives[mask,:]

            if len(d_nn_all_sub)>0:
                if hardest_sampling:
                    min_idx = torch.argmin(d_nn_all_sub)
                    z_nn_inside[i,:] = z_all_negatives_mask[min_idx,:]
                else:
                    min_idx = torch.randint(len(d_nn_all_sub),(1,))
                    z_nn_inside[i,:] = z_all_negatives_mask[min_idx,:]
            else:
                if hardest_sampling:
                    min_idx = torch.argmin(d_nn_all)
                    z_nn_inside[i,:] = z_all_negatives[min_idx,:]
                else:
                    min_idx = torch.randint(len(d_nn_all),(1,))
                    z_nn_inside[i,:] = z_all_negatives[min_idx,:]
           
        d_nn = ((z_a - z_nn_inside) ** 2).sum(dim=1)
        
        l_nn            = F.relu(d_pn                       + margin_nn - d_nn)  
        l_pn            = F.relu(d_pp_margin                + margin_pn - d_pn)
        l_pp            = (d_in_pp*margin_pp - d_pp)**2

        loss = beta_nn*l_nn + beta_pn*l_pn+ beta_pp*l_pp
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_a) + torch.norm(z_nn_inside) + torch.norm(z_pn) + beta_pp*torch.norm(z_pp))
        return loss

    def reconstruction_loss(self,x,x_pred,beta_traj=10.0):
        l = ((x - x_pred) ** 2).sum(dim=(2,3))
        l[:,0] = l[:,0]*beta_traj 
        l = l.sum(dim=(1))
        l = torch.mean(l)
        return l
    
    def reconstruction_loss_weighted(self,x,x_pred,beta_traj=1.0,beta_infra=1.0,beta_back_infra=1.0,beta_back_traj=1.0):
        traj_base = torch.clone(x[:,0,:,:])
        mask_traj = traj_base>0.0
        traj_pred = torch.clone(x_pred[:,0,:,:])
        traj_pred = torch.mul(traj_pred,mask_traj)

        infra_base = torch.clone(x[:,1,:,:])
        mask_infra = infra_base>0.0
        infra_pred = torch.clone(x_pred[:,1,:,:])
        infra_pred = torch.mul(infra_pred,mask_infra)
        
        mask_back_traj = traj_base==0.0
        back_traj = torch.clone(x_pred[:,0,:,:])
        back_traj = torch.mul(back_traj,mask_back_traj)

        mask_back_infra = infra_base==0.0
        back_infra = torch.clone(x_pred[:,1,:,:])
        back_infra = torch.mul(back_infra,mask_back_infra)

        l_traj = ((traj_base - traj_pred) ** 2).sum()/mask_traj.sum()
        l_infra = ((infra_base - infra_pred) ** 2).sum()/mask_infra.sum()
        l_back_traj = (back_traj ** 2).sum()/mask_back_traj.sum()
        l_back_infra = (back_infra **2).sum()/mask_back_infra.sum()
        l = beta_traj * l_traj + beta_infra * l_infra + beta_back_traj * l_back_traj + beta_back_infra * l_back_infra

        return l

    def loss(self, tuplets, margin_nn=1000.0,margin_pn=10.0,margin_pp=1.0, l2=0.0, beta_pp=0.0, beta_pn=0.0, beta_nn=1.0,alpha_recon=1.0, alpha_metric=1.0, beta_traj=1.0,beta_infra=1.0,beta_back_infra=1.0,beta_back_traj=1.0,hardest_sampling= True):
        """
        Computes loss for each batch.
        """
        z_a  = self.encode(tuplets['anchor_image'],tuplets['anchor_trajectory'])
        z_pp = self.encode(tuplets['pp_image'],tuplets['pp_trajectory'])
        z_pn = self.encode(tuplets['pn_image'],tuplets['pn_trajectory'])
        z_nn = self.encode(tuplets['nn_image'],tuplets['nn_trajectory'])
        anchor_pred = self.decode(z_a)
        
        loss_metric = self.metric_loss(z_a=z_a,
            z_pp=z_pp,
            z_pn=z_pn,
            z_nn=z_nn,
            d_in_pp=tuplets['pp_distance'],
            pn_valid=tuplets['pn_valid'],
            graphID_a=tuplets['anchor_group'],
            graphID_pp=tuplets['pp_group'],
            graphID_pn=tuplets['pn_group'],
            graphID_nn=tuplets['nn_group'],
            margin_nn=margin_nn,
            margin_pn=margin_pn,
            margin_pp=margin_pp,
            l2=l2,
            beta_pp=beta_pp,
            beta_pn=beta_pn,
            beta_nn=beta_nn,
            negatives_a = tuplets['negatives_anchor'],
            negatives_pp = tuplets['negatives_pp'],
            negatives_pn = tuplets['negatives_pn'],
            negatives_nn = tuplets['negatives_nn'],
            hardest_sampling = hardest_sampling
            )
        reconstruction_loss = self.reconstruction_loss_weighted(tuplets['anchor_target'], anchor_pred, beta_traj=beta_traj, beta_infra=beta_infra, beta_back_traj=beta_back_traj, beta_back_infra=beta_back_infra)
        loss = alpha_metric*loss_metric + alpha_recon*reconstruction_loss
        return loss,loss_metric,reconstruction_loss


def make_scenenet(encoderI_type=None, encoderT_type=None, merge_type=None, encoderI_args=None, encoderT_args=None, merge_args=None,z_dim_t=32,z_dim_i=64,z_dim_m=64,image_size=200,channels=1,traj_size=3):
    """
    Returns a SceneNet, providing a latent representation for road infrastructure images
    """
    return SceneNet(encoderI_type=encoderI_type, encoderT_type=encoderT_type, merge_type=merge_type, encoderI_args=encoderI_args, encoderT_args=encoderT_args, merge_args=merge_args,z_dim_t=z_dim_t,z_dim_i=z_dim_i,z_dim_m=z_dim_m,image_size=image_size,channels=channels,traj_size=traj_size)

