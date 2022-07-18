'''
----------------------------------------------------------------------
 BSD 3-Clause License

 Copyright (c) 2022, Jonas Wurst
 All rights reserved.
----------------------------------------------------------------------

'''
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize
import scipy.io
nn_sampling_types = ["random","group","random-excl"]

import mat73
import matplotlib
import time
import sys
from PIL import Image


image_size_mt_base = 400

class SceneTripletsDataset(Dataset):

    def __init__(self, scene_dir=None, input_augmentation=False,output_type='input',transform=None,nn_sampling=nn_sampling_types[1], sequence_length=30,orientation='north',image_size_px=200,image_size_mt=200,t_max=6.0):
        assert nn_sampling in nn_sampling_types, 'EncoderI keyword unknown'
        if not scene_dir:
            print('Data Folder not known')
            quit()

        self.scene_dir = scene_dir

        #mat = mat73.loadmat(scene_dir+ 'LUT.mat')
        #files = mat['files']

        mat = mat73.loadmat(scene_dir+ 'LUT.mat')
        files = mat['files']
        self.files = np.squeeze(files)

        files = mat['img_files']
        self.img_files = np.squeeze(files)

        bbox = mat['bbox']
        self.bbox = np.squeeze(bbox)

        heading = mat['heading']
        self.heading = np.squeeze(heading)

        #similars = mat['similars']
        #similars = np.squeeze(similars)#
        #self.similar_infra = similars[:,0]-1
        #self.similar_route = similars[:,0]-1
        #self.distance_route = similars[:,1]

        class_infra = mat['class_infra']
        self.class_infra = np.squeeze(class_infra).astype(int)
        #class_infra_container = mat['class_infra_container']
        #self.class_infra_container = class_infra_container
        #self.similar_infra_wo_route = []
        #for idx in range(self.class_infra.shape[0]):
        #    infra_class_id = self.class_infra[idx]-1
        #    self.similar_infra_wo_route.append(np.setdiff1d(np.asarray(self.class_infra_container[infra_class_id])-1,self.similar_route[idx]))
        
        group = mat['class']
        self.group = np.squeeze(group)

        #group_lables = np.unique(self.group)
        #group_dict = dict()
        #for label in group_lables:
        #    group_dict[label] = np.where(self.group == label)[0]
        #self.group_dict = group_dict

        #self.nn_sampling = nn_sampling
        #self.length = len(self.files)
        # Same group:
        # get all idx from same group 
        #self.nn_ids = []
        #if self.nn_sampling == nn_sampling_types[0]:
            # Random  : Sample from all but the the very same infrastructure
            #remaining_idx = np.arange(self.length)
            #for idx in range(self.class_infra.shape[0]):
            #    infra_class_id = self.class_infra[idx]-1
            #    self.nn_ids.append(np.setdiff1d(remaining_idx, np.asarray(self.class_infra_container[infra_class_id])-1))
        #    self.nn_ids =  np.arange(self.length)
        #elif self.nn_sampling == nn_sampling_types[1]:
            # Group : Sample from all in the same group but the the very same infrastructure
        #    for idx in range(self.class_infra.shape[0]):
        #        infra_class_id = self.class_infra[idx]-1
        #        remaining_idx = self.group_dict[self.group[idx]]
        #        self.nn_ids.append(np.setdiff1d(remaining_idx, np.asarray(self.class_infra_container[infra_class_id])-1))
        #elif self.nn_sampling == nn_sampling_types[2]:
            # Random excluding group
            #for idx in range(self.class_infra.shape[0]):
            #    possible_index = []
            #    for idx_inner,group_list in enumerate(self.group_dict):
            #        if idx_inner != self.group[idx]:
            #            possible_index.append(group_list)
            #    self.nn_ids.append(possible_index) 
        #    self.nn_ids = []
        #graph_size = mat['graphSize']
        #self.graph_size = np.squeeze(graph_size)
        self.transform = transform
        self.sequence_length = sequence_length
        self.image_size_px = image_size_px
        self.image_size_mt = image_size_mt
        self.image_res = float(image_size_mt/image_size_px)
        pos_norm = image_size_mt/2.0
        self.t_max = t_max
        if orientation=='north':
            # No rotation required
            # Load Traj
            self.trajectory = torch.zeros((len(self.files),self.sequence_length,3))     
            for idx,fileName in enumerate(self.files):
                mat = scipy.io.loadmat(fileName,variable_names=['pos', 'time'])
                x = torch.from_numpy(mat['pos'][:,0])
                self.trajectory[idx,0:x.shape[0],0] = (x- x[0])/pos_norm
                self.trajectory[idx,0:x.shape[0],1] = torch.from_numpy(mat['pos'][:,1]-mat['pos'][0,1])/pos_norm
                self.trajectory[idx,0:x.shape[0],2] = torch.from_numpy(mat['time'])/self.t_max

            # Load all images before
            self.image_data = torch.zeros((len(self.files),1,image_size_px,image_size_px),dtype=torch.uint8)     
            if self.image_size_mt < image_size_mt_base:
                min_px = image_size_mt_base/2-1 - image_size_mt/2
                max_px = image_size_mt_base/2-1 + image_size_mt/2

                for idx,fileName in enumerate(self.img_files):
                    image = Image.open(self.scene_dir + fileName).convert('L').crop((min_px,min_px,max_px,max_px))
                    image = image.resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data[idx,0,:,:] = image
            elif self.image_size_mt == image_size_mt_base:
                for idx,fileName in enumerate(self.img_files):
                    image = Image.open(self.scene_dir + fileName).convert('L').resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data[idx,0,:,:] = image
            else:
                print('Image size too large')
                quit()
        else:
            # No rotation required
            # Load Traj
            self.trajectory = torch.zeros((len(self.files),sequence_length,3))     
            for idx,fileName in enumerate(self.files):
                mat = scipy.io.loadmat(fileName,variable_names=['pos', 'time'])
                x = mat['pos'][:,0]-mat['pos'][0,0]
                y = mat['pos'][:,1]-mat['pos'][0,1]
                pos = np.vstack((x,y))
                theta = np.pi/2-self.heading[idx] 
                rot_mat = np.array(( (np.cos(theta), -np.sin(theta)),(np.sin(theta),  np.cos(theta)) ))
                pos = np.transpose(rot_mat.dot(pos))
                self.trajectory[idx,0:x.shape[0],0] = torch.from_numpy(pos[:,0])/pos_norm
                self.trajectory[idx,0:x.shape[0],1] = torch.from_numpy(pos[:,1])/pos_norm
                self.trajectory[idx,0:x.shape[0],2] = torch.from_numpy(mat['time'])/self.t_max

            # Load all images before
            self.image_data = torch.zeros((len(self.files),1,image_size_px,image_size_px),dtype=torch.uint8)     
            if self.image_size_mt < image_size_mt_base:
                min_px = image_size_mt_base/2-1 - image_size_mt/2
                max_px = image_size_mt_base/2-1 + image_size_mt/2

                for idx,fileName in enumerate(self.img_files):
                    theta = np.rad2deg(np.pi/2-self.heading[idx])
                    image = Image.open(self.scene_dir + fileName).convert('L').rotate(theta).crop((min_px,min_px,max_px,max_px))
                    image = image.resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data[idx,0,:,:] = image
            elif self.image_size_mt == image_size_mt_base:
                for idx,fileName in enumerate(self.img_files):
                    theta = np.rad2deg(np.pi/2-self.heading[idx])
                    image = Image.open(self.scene_dir + fileName).convert('L').rotate(theta).resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data[idx,0,:,:] = image
            else:
                print('Image size too large')
                quit()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Anchor---------------------------------------------------------------------------------------------------------- 
        anchor_image, anchor_trajectory, anchor_group, anchor_image_merged = self.get_image_trajectory(idx,merged_image_flag=True)
        sample = {'anchor_image': anchor_image, 'anchor_trajectory': anchor_trajectory, 'anchor_group': anchor_group, 'anchor_merged': anchor_image_merged}
        
        if self.transform:
            sample = self.transform(sample)

        
        return sample

    def get_image_trajectory(self,idx,merged_image_flag=False):
        image = self.image_data[idx,:,:,:].float()
        trajectory = self.trajectory[idx,:,:]
        group = self.group[idx]

        # Merged Image
        if merged_image_flag:

            trajectory_quant = torch.round(trajectory*self.image_size_px/2).type(torch.int64) 
            trajectory_quant[:,1] = -trajectory_quant[:,1]
            trajectory_quant = trajectory_quant+torch.tensor([self.image_size_px/2-1,self.image_size_px/2-1,0]).type(torch.int64) 
        
            image_merged = torch.zeros([2,self.image_size_px,self.image_size_px])
            image_merged[1,:,:] = image
        
            mask = (trajectory_quant[:,0]>(self.image_size_px-1)) | (trajectory_quant[:,1]>(self.image_size_px-1)) | (trajectory_quant[:,0]<0) | (trajectory_quant[:,1]<0)
        
            image_merged[0,trajectory_quant[~mask,1],trajectory_quant[~mask,0]] = trajectory[~mask,2]
        else:
            image_merged = torch.tensor([])


        return image, trajectory, group, image_merged

    


### TRANSFORMS ###
class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        sequence_length = 30
        a= sample['anchor_trajectory']
        pad_size = list(a.shape)
        pad_size[0] = sequence_length - a.size(0)
        a = torch.cat([a, torch.zeros(*pad_size)], dim=0)

        sample['anchor_trajectory'] = a
        return sample




def triplet_dataloader( scene_dir=None,orientation='north',input_augmentation=False,output_type='input',image_size_px=200,image_size_mt=200, batch_size=4, shuffle=False, num_workers=0,sequence_length=30,nn_sampling=nn_sampling_types[1]):
    """
    Returns a DataLoader with bw infrastructure images.
    """
    transform_list = []
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = SceneTripletsDataset(scene_dir=scene_dir,input_augmentation=input_augmentation,output_type=output_type,orientation=orientation,image_size_px=image_size_px,image_size_mt=image_size_mt,transform=transform,sequence_length=sequence_length,nn_sampling=nn_sampling)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers,pin_memory=True)
    return dataloader

    