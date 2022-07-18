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

class FC(nn.Module):
    def __init__(self,n_layers,dims):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.add_module("Lin_"+str(i),nn.Linear(dims[i][0],dims[i][1]))
            if i== n_layers-1:
                break
            self.layers.add_module("Activation_"+str(i),nn.ReLU())
    def forward(self, z):
        x = self.layers(z)
        return x

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
        '''
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
        '''
        dims = []

        n_layers = 4
        dims = [[z_dim_m,704]]
        for i in range(n_layers-2):
                    dims.append([704,704])
        dims.append([704,704])
        self.classifier_1 = FC(n_layers=n_layers,dims=dims)

        dims = [[z_dim_m,2330]]
        for i in range(n_layers-2):
                    dims.append([2330,2330])
        dims.append([2330,2330])

        self.classifier_2 = FC(n_layers=n_layers,dims=dims)
        

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
    
    def forward_classification(self,z):
        class_1 = self.classifier_1.forward(z)
        class_2 = self.classifier_2.forward(z)

        return class_1,class_2



    def forward(self, x_image,x_traj):
        z = self.encode(x_image,x_traj)
        class_1,class_2 = self.forward_classification(z)
        #x = self.decode(z)
        return class_1,class_2


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

    def loss(self, tuplets):
        """
        Computes loss for each batch.
        """
        z_a  = self.encode(tuplets['anchor_image'],tuplets['anchor_trajectory'])
        loss_temp = torch.nn.CrossEntropyLoss()
        class_1,class_2 = self.forward_classification(z_a)
        
        loss_class_1 = loss_temp(class_1,tuplets['anchor_class_1'])
        loss_class_2 = loss_temp(class_2,tuplets['anchor_class_2'])
        #reconstruction_loss = self.reconstruction_loss_weighted(tuplets['anchor_target'], anchor_pred, beta_traj=beta_traj, beta_infra=beta_infra, beta_back_traj=beta_back_traj, beta_back_infra=beta_back_infra)
        loss = loss_class_1+loss_class_2
        return loss


def make_scenenet(encoderI_type=None, encoderT_type=None, merge_type=None, encoderI_args=None, encoderT_args=None, merge_args=None,z_dim_t=32,z_dim_i=64,z_dim_m=64,image_size=200,channels=1,traj_size=3):
    """
    Returns a SceneNet, providing a latent representation for road infrastructure images
    """
    return SceneNet(encoderI_type=encoderI_type, encoderT_type=encoderT_type, merge_type=merge_type, encoderI_args=encoderI_args, encoderT_args=encoderT_args, merge_args=merge_args,z_dim_t=z_dim_t,z_dim_i=z_dim_i,z_dim_m=z_dim_m,image_size=image_size,channels=channels,traj_size=traj_size)

