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
import scipy.io

nn_sampling_types = ["random","group","random-excl"]

import mat73
from PIL import Image


image_size_mt_base = 400

class SceneTripletsDataset(Dataset):

    def __init__(self, scene_dir=None,input_augmentation=False,output_type='input', transform=None,nn_sampling=nn_sampling_types[1], sequence_length=30,orientation='north',image_size_px=200,image_size_mt=200,t_max=6.0):
        assert nn_sampling in nn_sampling_types, 'EncoderI keyword unknown'
        if not scene_dir:
            print('Data Folder not known')
            quit()

        self.scene_dir = scene_dir
        mat = mat73.loadmat(scene_dir+ 'LUT.mat')
        files = mat['files']
        self.files = np.squeeze(files)

        files = mat['img_files']
        self.img_files = np.squeeze(files)

        heading = mat['heading']
        self.heading = np.squeeze(heading)

        similars = mat['similars']
        similars = np.squeeze(similars) # Causing the warning but works porperly
        self.similar_route = similars[:,0]-1
        self.distance_route = similars[:,1]

        # Get the per graph class 1...>1k
        class_infra = mat['class_infra']
        self.class_infra = np.squeeze(class_infra).astype(int)-1
        class_infra_container_in = mat['class_infra_container']
        class_infra_container = [np.array(class_infra_container_sub)-1 for class_infra_container_sub in class_infra_container_in]
        self.class_infra_container = class_infra_container

        # Get the rough class 1...8
        group = mat['class']
        self.group = np.squeeze(group)-1

        group_lables = np.unique(self.group)
        group_dict = dict()
        for label in group_lables:
            group_dict[label] = np.where(self.group == label)[0]
        self.group_dict = group_dict


        self.similar_infra_wo_route = []
        for idx in range(self.class_infra.shape[0]):
            infra_class_id = self.class_infra[idx]
            temp_diff = np.setdiff1d(np.asarray(self.class_infra_container[infra_class_id]),self.similar_route[idx])
            if temp_diff.size == 0:
                temp_diff = np.setdiff1d(self.group_dict[self.group[idx]],self.similar_route[idx])
            self.similar_infra_wo_route.append(temp_diff)
        
        

        self.nn_sampling = nn_sampling
        self.length = len(self.files)
        self.nn_ids = []
        if self.nn_sampling == nn_sampling_types[0]:
            # Random  : Sample from all but the the very same infrastructure
            self.nn_ids =  np.arange(self.length)
        elif self.nn_sampling == nn_sampling_types[1]:
            # Group : Sample from all in the same group but the the very same infrastructure
            #TODO index issue??
            for idx in range(self.class_infra.shape[0]):
                infra_class_id = self.class_infra[idx]
                remaining_idx = self.group_dict[self.group[idx]]
                self.nn_ids.append(np.setdiff1d(remaining_idx, np.asarray(self.class_infra_container[infra_class_id])))
        elif self.nn_sampling == nn_sampling_types[2]:
            # Random excluding group
            self.nn_ids = []

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
            self.trajectory = torch.zeros((similars.shape[0],sequence_length,3))     
            for idx,fileName in enumerate(self.files):
                mat = scipy.io.loadmat(fileName,variable_names=['pos', 'time'])
                x = torch.from_numpy(mat['pos'][:,0])
                self.trajectory[idx,0:x.shape[0],0] = (x- x[0])/pos_norm
                self.trajectory[idx,0:x.shape[0],1] = torch.from_numpy(mat['pos'][:,1]-mat['pos'][0,1])/pos_norm
                self.trajectory[idx,0:x.shape[0],2] = torch.from_numpy(mat['time'])/self.t_max

            # Load all images before
            self.image_data = torch.zeros((similars.shape[0],1,image_size_px,image_size_px),dtype=torch.uint8)     
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
            self.trajectory = torch.zeros((similars.shape[0],sequence_length,3))     
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
            self.image_data = torch.zeros((similars.shape[0],1,image_size_px,image_size_px),dtype=torch.uint8)     
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
        
        if input_augmentation:
            #TODO
            print('Not implemented')
        # ===========================================================================================================================
        # OUTPUT TARGET--------------------------------
        #=============================================================================================================================
        self.output_type = output_type
        if output_type =='input':
            self.trajectory_out = []
            self.image_data_out = []
        elif output_type =='north_noaug':
            self.trajectory_out = torch.zeros((similars.shape[0],sequence_length,3))     
            for idx,fileName in enumerate(self.files):
                mat = scipy.io.loadmat(fileName,variable_names=['pos', 'time'])
                x = torch.from_numpy(mat['pos'][:,0])
                self.trajectory_out[idx,0:x.shape[0],0] = (x- x[0])/pos_norm
                self.trajectory_out[idx,0:x.shape[0],1] = torch.from_numpy(mat['pos'][:,1]-mat['pos'][0,1])/pos_norm
                self.trajectory_out[idx,0:x.shape[0],2] = torch.from_numpy(mat['time'])/self.t_max

            # Load all images before
            self.image_data_out = torch.zeros((similars.shape[0],1,image_size_px,image_size_px),dtype=torch.uint8)     
            if self.image_size_mt < image_size_mt_base:
                min_px = image_size_mt_base/2-1 - image_size_mt/2
                max_px = image_size_mt_base/2-1 + image_size_mt/2

                for idx,fileName in enumerate(self.img_files):
                    fn_temp = fileName
                    image = Image.open(fn_temp).convert('L').crop((min_px,min_px,max_px,max_px))
                    image = image.resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data_out[idx,0,:,:] = image
            elif self.image_size_mt == image_size_mt_base:
                for idx,fileName in enumerate(self.img_files):
                    fn_temp = fileName
                    image = Image.open(fn_temp).convert('L').resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data_out[idx,0,:,:] = image
            else:
                print('Image size too large')
                quit()
        elif output_type =='north_aug':
            self.trajectory_out = torch.zeros((similars.shape[0],sequence_length,3))     
            for idx,fileName in enumerate(self.files):
                mat = scipy.io.loadmat(fileName,variable_names=['pos', 'time'])
                x = torch.from_numpy(mat['pos'][:,0])
                self.trajectory_out[idx,0:x.shape[0],0] = (x- x[0])/pos_norm
                self.trajectory_out[idx,0:x.shape[0],1] = torch.from_numpy(mat['pos'][:,1]-mat['pos'][0,1])/pos_norm
                self.trajectory_out[idx,0:x.shape[0],2] = torch.from_numpy(mat['time'])/self.t_max

            # Load all images before
            self.image_data_out = torch.zeros((similars.shape[0],1,image_size_px,image_size_px),dtype=torch.uint8)     
            if self.image_size_mt < image_size_mt_base:
                min_px = image_size_mt_base/2-1 - image_size_mt/2
                max_px = image_size_mt_base/2-1 + image_size_mt/2

                for idx,fileName in enumerate(self.files):
                    fn_temp = fileName[:-6]+'reach.png'
                    image = Image.open(fn_temp).convert('L').crop((min_px,min_px,max_px,max_px))
                    image = image.resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data_out[idx,0,:,:] = image
            elif self.image_size_mt == image_size_mt_base:
                for idx,fileName in enumerate(self.files):
                    fn_temp = fileName[:-6]+'reach.png'
                    image = Image.open(fn_temp).convert('L').resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data_out[idx,0,:,:] = image
            else:
                print('Image size too large')
                quit()
        elif output_type =='ego_noaug':
            self.trajectory_out = torch.zeros((similars.shape[0],sequence_length,3))     
            for idx,fileName in enumerate(self.files):
                mat = scipy.io.loadmat(fileName,variable_names=['pos', 'time'])
                x = mat['pos'][:,0]-mat['pos'][0,0]
                y = mat['pos'][:,1]-mat['pos'][0,1]
                pos = np.vstack((x,y))
                theta = np.pi/2-self.heading[idx] 
                rot_mat = np.array(( (np.cos(theta), -np.sin(theta)),(np.sin(theta),  np.cos(theta)) ))
                pos = np.transpose(rot_mat.dot(pos))
                self.trajectory_out[idx,0:x.shape[0],0] = torch.from_numpy(pos[:,0])/pos_norm
                self.trajectory_out[idx,0:x.shape[0],1] = torch.from_numpy(pos[:,1])/pos_norm
                self.trajectory_out[idx,0:x.shape[0],2] = torch.from_numpy(mat['time'])/self.t_max

            # Load all images before
            self.image_data_out = torch.zeros((similars.shape[0],1,image_size_px,image_size_px),dtype=torch.uint8)     
            if self.image_size_mt < image_size_mt_base:
                min_px = image_size_mt_base/2-1 - image_size_mt/2
                max_px = image_size_mt_base/2-1 + image_size_mt/2

                for idx,fileName in enumerate(self.img_files):
                    theta = np.rad2deg(np.pi/2-self.heading[idx])
                    image = Image.open(self.scene_dir + fileName).convert('L').rotate(theta).crop((min_px,min_px,max_px,max_px))
                    image = image.resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data_out[idx,0,:,:] = image
            elif self.image_size_mt == image_size_mt_base:
                for idx,fileName in enumerate(self.img_files):
                    theta = np.rad2deg(np.pi/2-self.heading[idx])
                    image = Image.open(self.scene_dir + fileName).convert('L').rotate(theta).resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data_out[idx,0,:,:] = image
            else:
                print('Image size too large')
                quit()
        elif output_type =='ego_aug':
            self.trajectory_out = torch.zeros((similars.shape[0],sequence_length,3))     
            for idx,fileName in enumerate(self.files):
                mat = scipy.io.loadmat(fileName,variable_names=['pos', 'time'])
                x = mat['pos'][:,0]-mat['pos'][0,0]
                y = mat['pos'][:,1]-mat['pos'][0,1]
                pos = np.vstack((x,y))
                theta = np.pi/2-self.heading[idx] 
                rot_mat = np.array(( (np.cos(theta), -np.sin(theta)),(np.sin(theta),  np.cos(theta)) ))
                pos = np.transpose(rot_mat.dot(pos))
                self.trajectory_out[idx,0:x.shape[0],0] = torch.from_numpy(pos[:,0])/pos_norm
                self.trajectory_out[idx,0:x.shape[0],1] = torch.from_numpy(pos[:,1])/pos_norm
                self.trajectory_out[idx,0:x.shape[0],2] = torch.from_numpy(mat['time'])/self.t_max

            # Load all images before
            self.image_data_out = torch.zeros((similars.shape[0],1,image_size_px,image_size_px),dtype=torch.uint8)     
            if self.image_size_mt < image_size_mt_base:
                min_px = image_size_mt_base/2-1 - image_size_mt/2
                max_px = image_size_mt_base/2-1 + image_size_mt/2

                for idx,fileName in enumerate(self.files):
                    fn_temp = fileName[:-6]+'reach.png'
                    theta = np.rad2deg(np.pi/2-self.heading[idx])
                    image = Image.open(fn_temp).convert('L').rotate(theta).crop((min_px,min_px,max_px,max_px))
                    image = image.resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data_out[idx,0,:,:] = image
            elif self.image_size_mt == image_size_mt_base:
                for idx,fileName in enumerate(self.files):
                    fn_temp = fileName[:-6]+'reach.png'
                    theta = np.rad2deg(np.pi/2-self.heading[idx])
                    image = Image.open(fn_temp).convert('L').rotate(theta).resize((image_size_px,image_size_px))
                    image = transforms.ToTensor()(image)
                    image = image >0
                    self.image_data_out[idx,0,:,:] = image
            else:
                print('Image size too large')
                quit()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Anchor---------------------------------------------------------------------------------------------------------- 
        anchor_image, anchor_trajectory, anchor_group, anchor_image_merged = self.get_image_trajectory(idx,merged_image_flag=True)
        sample = {'anchor_image': anchor_image, 'anchor_trajectory': anchor_trajectory, 'anchor_group': anchor_group, 'anchor_merged': anchor_image_merged}
        
        if self.output_type == 'input':
            sample['anchor_target'] = anchor_image_merged
        else:
            image = self.image_data_out[idx,:,:,:].float()
            trajectory = self.trajectory_out[idx,:,:]
            trajectory_quant = torch.round(trajectory*self.image_size_px/2).type(torch.int64) 
            trajectory_quant[:,1] = -trajectory_quant[:,1]
            trajectory_quant = trajectory_quant+torch.tensor([self.image_size_px/2-1,self.image_size_px/2-1,0]).type(torch.int64) 
            image_merged = torch.zeros([2,self.image_size_px,self.image_size_px])
            image_merged[1,:,:] = image
            mask = (trajectory_quant[:,0]>(self.image_size_px-1)) | (trajectory_quant[:,1]>(self.image_size_px-1)) | (trajectory_quant[:,0]<0) | (trajectory_quant[:,1]<0)
            image_merged[0,trajectory_quant[~mask,1],trajectory_quant[~mask,0]] = trajectory[~mask,2]
            sample['anchor_target'] = image_merged
            
        # PP Positive infra Positive Route-------------------------------------------------------------------------------
        possible_index = self.similar_route[idx].astype(int)#[0]
        idx_neighbor = idx
        while idx_neighbor == idx:
            idx_neighbor_idx = torch.randint(possible_index.size,(1,))
            idx_neighbor = possible_index[np.squeeze(idx_neighbor_idx.numpy()).astype(int)]

        pp_image, pp_trajectory, pp_group, pp_image_merged = self.get_image_trajectory(idx_neighbor)
        pp_distance = self.distance_route[idx][np.squeeze(idx_neighbor_idx.numpy()).astype(int)]
        sample['pp_image'] = pp_image
        sample['pp_trajectory'] = pp_trajectory
        sample['pp_group'] = pp_group
        sample['pp_distance'] = pp_distance
        sample['pp_idx'] = idx_neighbor

        # PN Positive infra Negative Route----------------------------------------------------------------------------
        possible_index = self.similar_infra_wo_route[idx].astype(int)
        if possible_index.size == 0:
            pn_image = pp_image
            pn_trajectory = pp_trajectory
            pn_group = pp_group
            pn_valid = 0.0
            sample['pn_idx'] = sample['pp_idx']
            #print('a',possible_index.size)
        else:
            idx_neighbor = idx
            while idx_neighbor == idx:
                idx_neighbor_idx = torch.randint(possible_index.size,(1,))
                idx_neighbor = possible_index[np.squeeze(idx_neighbor_idx.numpy()).astype(int)]

            pn_image, pn_trajectory, pn_group, pn_image_merged = self.get_image_trajectory(idx_neighbor)
            pn_valid = 1.0
            sample['pn_idx'] = idx_neighbor
        
        sample['pn_image'] = pn_image
        sample['pn_trajectory'] = pn_trajectory
        sample['pn_group'] = pn_group
        sample['pn_valid'] = pn_valid

        
        # NN Neagtive infra Negative Route----------------------------------------------------------------------------

        if self.nn_sampling == nn_sampling_types[0]:
            # Random  : Sample from all but the the very same infrastructure
            infra_class_id = self.class_infra[idx]
            possible_index = self.nn_ids
            index_temp = np.asarray(self.class_infra_container[infra_class_id]).astype(int)
            possible_index = np.delete(possible_index,index_temp)
        elif self.nn_sampling == nn_sampling_types[1]:
            # Group : Sample from all in the same group but the the very same infrastructure
            possible_index = self.nn_ids[idx]
        elif self.nn_sampling == nn_sampling_types[2]:
            # Random excluding group
            possible_index = np.array([])
            for idx_inner,(group_id,group_list) in enumerate(self.group_dict.items()):
                if group_id != self.group[idx]:
                    possible_index = np.append(possible_index,group_list)
            possible_index =possible_index.astype(int)
            
        
        
        if possible_index.size == 0:
            # Handle the empty case
            nn_image = pn_image
            nn_trajectory = pn_trajectory
            nn_group = pn_group
            sample['nn_idx'] = sample['pn_idx']
        else:
            idx_neighbor = idx
            while idx_neighbor == idx:
                idx_neighbor_idx = torch.randint(possible_index.size,(1,))
                idx_neighbor = possible_index[np.squeeze(idx_neighbor_idx.numpy()).astype(int)]
            
            nn_image, nn_trajectory, nn_group, nn_image_merged = self.get_image_trajectory(idx_neighbor)
            sample['nn_idx'] = idx_neighbor
        
        sample['nn_image'] = nn_image
        sample['nn_trajectory'] = nn_trajectory
        sample['nn_group'] = nn_group
        

        sample['idx'] = idx

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
    
    def get_negatives_in_batch(self,tuplets):

        idx_list_anchor = tuplets['idx']
        idx_list_pp = tuplets['pp_idx']
        idx_list_pn = tuplets['pn_idx']
        idx_list_nn = tuplets['nn_idx']

        # Over each batch element 
        negatives_anchor = []
        negatives_pp = []
        negatives_pn = []
        negatives_nn = []
        for idx in idx_list_anchor:
            # if random : check for any other class_infra
            # if group. check for any other class_infra within the same group
            # if rand excl. check for any other group
            idx_list = idx_list_anchor
            if self.nn_sampling == nn_sampling_types[0]:
                # Random
                batch_idx = self.class_infra[idx_list]!=self.class_infra[idx]
            elif self.nn_sampling == nn_sampling_types[1]:
                # Group
                batch_idx = (self.class_infra[idx_list]!=self.class_infra[idx]) & (self.group[idx_list]==self.group[idx])
            elif self.nn_sampling == nn_sampling_types[2]:
                # Random excld    
                batch_idx = self.group[idx_list]!=self.group[idx]
            negatives_anchor.append(batch_idx)  

            idx_list = idx_list_pp
            if self.nn_sampling == nn_sampling_types[0]:
                # Random
                batch_idx = self.class_infra[idx_list]!=self.class_infra[idx]
            elif self.nn_sampling == nn_sampling_types[1]:
                # Group
                batch_idx = (self.class_infra[idx_list]!=self.class_infra[idx]) & (self.group[idx_list]==self.group[idx])
            elif self.nn_sampling == nn_sampling_types[2]:
                # Random excld    
                batch_idx = self.group[idx_list]!=self.group[idx]
            negatives_pp.append(batch_idx)

            idx_list = idx_list_pn
            if self.nn_sampling == nn_sampling_types[0]:
                # Random
                batch_idx = self.class_infra[idx_list]!=self.class_infra[idx]
            elif self.nn_sampling == nn_sampling_types[1]:
                # Group
                batch_idx = (self.class_infra[idx_list]!=self.class_infra[idx]) & (self.group[idx_list]==self.group[idx])
            elif self.nn_sampling == nn_sampling_types[2]:
                # Random excld    
                batch_idx = self.group[idx_list]!=self.group[idx]
            negatives_pn.append(batch_idx)

            idx_list = idx_list_nn
            if self.nn_sampling == nn_sampling_types[0]:
                # Random
                batch_idx = self.class_infra[idx_list]!=self.class_infra[idx]
            elif self.nn_sampling == nn_sampling_types[1]:
                # Group
                batch_idx = (self.class_infra[idx_list]!=self.class_infra[idx]) & (self.group[idx_list]==self.group[idx])
            elif self.nn_sampling == nn_sampling_types[2]:
                # Random excld    
                batch_idx = self.group[idx_list]!=self.group[idx]
            negatives_nn.append(batch_idx)

        tuplets['negatives_anchor'] = torch.BoolTensor(negatives_anchor)
        tuplets['negatives_pp'] = torch.BoolTensor(negatives_pp)
        tuplets['negatives_pn'] = torch.BoolTensor(negatives_pn)
        tuplets['negatives_nn'] = torch.BoolTensor(negatives_nn)

        return tuplets
    


### TRANSFORMS ###
class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        # TODO check if needed
        sequence_length = 30
        a, pp, pn, nn = (sample['anchor_trajectory'],sample['pp_trajectory'],sample['pn_trajectory'],sample['nn_trajectory'])
        pad_size = list(a.shape)
        pad_size[0] = sequence_length - a.size(0)
        a = torch.cat([a, torch.zeros(*pad_size)], dim=0)

        pad_size[0] = sequence_length - pp.size(0)
        pp = torch.cat([pp, torch.zeros(*pad_size)], dim=0)

        pad_size[0] = sequence_length - pn.size(0)
        pn = torch.cat([pn, torch.zeros(*pad_size)], dim=0)

        pad_size[0] = sequence_length - nn.size(0)
        nn = torch.cat([nn, torch.zeros(*pad_size)], dim=0)
        sample['anchor_trajectory'] = a
        sample['pp_trajectory'] = pp
        sample['pn_trajectory'] = pn
        sample['nn_trajectory'] = nn

        return sample




def triplet_dataloader( scene_dir=None,orientation='north',input_augmentation=False,output_type='input',image_size_px=200,image_size_mt=200, batch_size=4, shuffle=True, num_workers=0,nn_sampling=nn_sampling_types[1],sequence_length=30):
    """
    Returns a DataLoader with bw infrastructure images.
    """
    transform_list = []
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = SceneTripletsDataset(scene_dir=scene_dir,input_augmentation=input_augmentation,output_type=output_type,orientation=orientation,image_size_px=image_size_px,image_size_mt=image_size_mt,transform=transform,nn_sampling=nn_sampling,sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers,pin_memory=True)
    return dataloader

    