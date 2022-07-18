'''
----------------------------------------------------------------------
 BSD 3-Clause License

 Copyright (c) 2022, Jonas Wurst
 All rights reserved.
----------------------------------------------------------------------

'''
import torch
from torch import nn
from src.vit_pytorch import ViT
from src.traj_encoder import traj_encoder
import torchvision.models as models

encoderI_types = ["ViT","ResNet-18"]
encoderT_types = ["LSTM","Transformer-Encoder"]
merge_types =["FC"]

class FC(nn.Module):
    def __init__(self,n_layers,dims):
        super().__init__()
        self.layers = nn.Sequential()
        for i in range(n_layers):
            self.layers.add_module("Lin_"+str(i),nn.Linear(dims[i][0],dims[i][1]))
            if i== n_layers-1:
                break
            self.layers.add_module("Activation_"+str(i),nn.ReLU())
    def forward(self, z_image,z_traj):
        x = torch.cat((z_image,z_traj),1)
        x = self.layers(x)
        return x

class EncoderFull(nn.Module):
    def __init__(self, *, encoderI_type=encoderI_types[0], encoderT_type=encoderT_types[0], merge_type=merge_types[0], encoderI_args=None, encoderT_args=None, merge_args=None,z_dim_t=32,z_dim_i=64,z_dim_m=64,image_size=200,channels=1,traj_size=3):
        super().__init__()
        assert encoderI_type in encoderI_types, 'EncoderI keyword unknown'
        assert encoderT_type in encoderT_types, 'EncoderT keyword unknown'
        assert merge_type in merge_types, 'Merger keyword unknown'
        self.encoderI_type = encoderI_type
        self.encoderT_type = encoderT_type
        self.merge_type = merge_type

        # Image Encoder==========================================================================================================================
        if self.encoderI_type == encoderI_types[0]:
            #---------------------------------------
            #ViT
            #---------------------------------------
            if not encoderI_args:
                self.encoder_image = ViT(image_size = image_size,channels=channels,z_dim = z_dim_i,patch_size = 20,dim = 256,depth = 10,heads = 16,mlp_dim = 128)
            else:
                self.encoder_image = ViT(image_size = image_size,channels=channels,z_dim = z_dim_i,**encoderI_args)
        elif self.encoderI_type == encoderI_types[1]:
            #---------------------------------------
            #ResNet-18
            #---------------------------------------
            self.encoder_image  = models.resnet18()
            self.encoder_image.conv1 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder_image.fc = nn.Linear(512,z_dim_i, bias=True)
        
        # Trajectory Encoder==========================================================================================================================
        if self.encoderT_type == encoderT_types[0]:
            #---------------------------------------
            #LSTM
            #---------------------------------------
            self.encoder_trajectory = nn.LSTM(input_size = traj_size,hidden_size = z_dim_t,batch_first =True)
        elif self.encoderT_type == encoderT_types[1]:
            #---------------------------------------
            #Transformer
            #---------------------------------------
            if not encoderT_args:
                self.encoder_trajectory = traj_encoder(dim_in = traj_size, dim_out=z_dim_t, depth=6, heads=8, dim_trans=64, dim_mlp=128)
            else:
                self.encoder_trajectory = traj_encoder(dim_in = traj_size, dim_out=z_dim_t, **encoderT_args)
        

        # Encoder Merger==========================================================================================================================
        if self.merge_type == merge_types[0]:
            #---------------------------------------
            # FC      
            #---------------------------------------
            dims = [[z_dim_t+z_dim_i,z_dim_t+z_dim_i]]
            if not merge_args:
                n_layers = 2
                for i in range(n_layers-2):
                    dims.append([z_dim_t+z_dim_i,z_dim_t+z_dim_i])
                dims.append([z_dim_t+z_dim_i,z_dim_m])
                self.encoder_merge = FC(n_layers=n_layers,dims=dims)
            else:
                n_layers = merge_args['n_layers']
                for i in range(n_layers-2):
                    dims.append([z_dim_t+z_dim_i,z_dim_t+z_dim_i])
                dims.append([z_dim_t+z_dim_i,z_dim_m])
                self.encoder_merge = FC(dims=dims,**merge_args)

    def forward(self,x_image,x_traj):
        z_image = self.encoder_image.forward(x_image)
        first_element = ((x_traj[:,1:,2]==0)==0).sum(dim=1)+1 
        packed_trajectory = torch.nn.utils.rnn.pack_padded_sequence(x_traj, first_element.cpu().numpy(), batch_first=True, enforce_sorted=False)
        z_traj = self.encoder_trajectory.forward(packed_trajectory)
        if self.encoderT_type == encoderT_types[0]:
            z_traj = z_traj[1][0][0,:,:]        
        #z_traj = torch.zeros_like(z_traj)
        z_merged = self.encoder_merge.forward(z_image,z_traj)
        return z_merged
