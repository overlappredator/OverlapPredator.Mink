"""
Author: Shengyu Huang
Last modified: 30.11.2020
"""

import os,sys,glob,torch
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import open3d as o3d
from lib.benchmark_utils import to_o3d_pcd, to_tsfm, get_correspondences, to_tensor
import MinkowskiEngine as ME

class IndoorDataset(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,infos,config,data_augmentation=True):
        super(IndoorDataset,self).__init__()
        self.infos = infos
        self.base_dir = config.root
        self.overlap_radius = config.overlap_radius
        self.data_augmentation=data_augmentation
        self.config = config
        self.voxel_size = config.voxel_size
        self.search_voxel_size = self.voxel_size * 1.5
        
        self.rot_factor=1.
        self.augment_noise = config.augment_noise

    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self,item): 
        # get transformation
        rot=self.infos['rot'][item]
        trans=self.infos['trans'][item]

        # get pointcloud
        src_path=os.path.join(self.base_dir,self.infos['src'][item])
        tgt_path=os.path.join(self.base_dir,self.infos['tgt'][item])
        src_pcd = torch.load(src_path)
        tgt_pcd = torch.load(tgt_path)

        # add gaussian noise
        if self.data_augmentation:            
            # rotate the point cloud
            euler_ab=np.random.rand(3)*np.pi*2/self.rot_factor # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            if(np.random.rand(1)[0]>0.5):
                src_pcd=np.matmul(rot_ab,src_pcd.T).T
                rot=np.matmul(rot,rot_ab.T)
            else:
                tgt_pcd=np.matmul(rot_ab,tgt_pcd.T).T
                rot=np.matmul(rot_ab,rot)
                trans=np.matmul(rot_ab,trans)

            src_pcd += (np.random.rand(src_pcd.shape[0],3) - 0.5) * self.augment_noise
            tgt_pcd += (np.random.rand(tgt_pcd.shape[0],3) - 0.5) * self.augment_noise
        
        if(trans.ndim==1):
            trans=trans[:,None]

        # build sparse tensor
        _, sel_src = ME.utils.sparse_quantize(np.ascontiguousarray(src_pcd) / self.voxel_size, return_index=True)
        _, sel_tgt = ME.utils.sparse_quantize(np.ascontiguousarray(tgt_pcd) / self.voxel_size, return_index=True)

        # get correspondence
        tsfm = to_tsfm(rot, trans)
        src_xyz, tgt_xyz = src_pcd[sel_src], tgt_pcd[sel_tgt] # raw point clouds
        matching_inds = get_correspondences(to_o3d_pcd(src_xyz), to_o3d_pcd(tgt_xyz), tsfm, self.search_voxel_size)

        # get voxelized coordinates
        src_coords, tgt_coords = np.floor(src_xyz / self.voxel_size), np.floor(tgt_xyz / self.voxel_size)

        # get feats
        src_feats = np.ones((src_coords.shape[0],1),dtype=np.float32)
        tgt_feats = np.ones((tgt_coords.shape[0],1),dtype=np.float32)

        src_xyz, tgt_xyz = to_tensor(src_xyz).float(), to_tensor(tgt_xyz).float()
        rot, trans = to_tensor(rot), to_tensor(trans)

        return src_xyz, tgt_xyz, src_coords, tgt_coords, src_feats, tgt_feats, matching_inds, rot, trans
