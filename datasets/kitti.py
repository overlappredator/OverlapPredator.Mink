# Basic libs
import os, time, glob, random, pickle, copy, torch
import numpy as np
import open3d
from scipy.spatial.transform import Rotation
import MinkowskiEngine as ME

# Dataset parent class
from torch.utils.data import Dataset
from lib.benchmark_utils import to_tsfm, to_o3d_pcd, get_correspondences, to_tensor


class KITTIDataset(Dataset):
    """
    We augment data with rotation, scaling, and translation
    Then we get correspondence, and voxelise them, 
    """
    DATA_FILES = {
        'train': './configs/kitti/train_kitti.txt',
        'val': './configs/kitti/val_kitti.txt',
        'test': './configs/kitti/test_kitti.txt'
    }
    def __init__(self,config,split,data_augmentation=True):
        super(KITTIDataset,self).__init__()
        self.config = config
        self.root = os.path.join(config.root,'dataset')
        self.icp_path = os.path.join(config.root,'icp')
        if not os.path.exists(self.icp_path):
            os.makedirs(self.icp_path)
        self.voxel_size = config.voxel_size
        self.search_voxel_size = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.augment_noise = config.augment_noise
        self.IS_ODOMETRY = True
        self.augment_shift_range = config.augment_shift_range
        self.augment_scale_max = config.augment_scale_max
        self.augment_scale_min = config.augment_scale_min
        self.max_corr = config.max_points

        # Initiate containers
        self.files = []
        self.kitti_icp_cache = {}
        self.kitti_cache = {}
        self.prepare_kitti_ply(split)
        self.split = split


    def prepare_kitti_ply(self, split):
        assert split in ['train','val','test']

        subset_names = open(self.DATA_FILES[split]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + '/sequences/%02d/velodyne/*.bin' % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])

            # get one-to-one distance by comparing the translation vector
            all_odo = self.get_video_odometry(drive_id, return_all=True)
            all_pos = np.array([self.odometry_to_positions(odo) for odo in all_odo])
            Ts = all_pos[:, :3, 3]
            pdist = (Ts.reshape(1, -1, 3) - Ts.reshape(-1, 1, 3)) ** 2
            pdist = np.sqrt(pdist.sum(-1)) 

            ######################################
            # D3Feat script to generate test pairs
            more_than_10 = pdist > 10
            curr_time = inames[0]
            while curr_time in inames:
                next_time = np.where(more_than_10[curr_time][curr_time:curr_time + 100])[0]
                if len(next_time) == 0:
                    curr_time += 1
                else:
                    next_time = next_time[0] + curr_time - 1

                if next_time in inames:
                    self.files.append((drive_id, curr_time, next_time))
                    curr_time = next_time + 1

        # remove bad pairs
        if split=='test':
            self.files.remove((8, 15, 58))
        print(f'Num_{split}: {len(self.files)}')



    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):
        drive = self.files[idx][0]
        t0, t1 = self.files[idx][1], self.files[idx][2]
        all_odometry = self.get_video_odometry(drive, [t0, t1])
        positions = [self.odometry_to_positions(odometry) for odometry in all_odometry]
        fname0 = self._get_velodyne_fn(drive, t0)
        fname1 = self._get_velodyne_fn(drive, t1)

        # extract xyz
        xyz0 = np.fromfile(fname0, dtype=np.float32).reshape(-1, 4)[:,:3]
        xyz1 = np.fromfile(fname1, dtype=np.float32).reshape(-1, 4)[:,:3]

        # use ICP to refine the ground_truth pose, for ICP we don't voxllize the point clouds
        key = '%d_%d_%d' % (drive, t0, t1)
        filename = self.icp_path + '/' + key + '.npy'
        if key not in self.kitti_icp_cache:
            if not os.path.exists(filename):
                print('missing ICP files, recompute it')
                M = (self.velo2cam @ positions[0].T @ np.linalg.inv(positions[1].T)
                            @ np.linalg.inv(self.velo2cam)).T
                xyz0_t = self.apply_transform(xyz0, M)
                pcd0 = to_o3d_pcd(xyz0_t)
                pcd1 = to_o3d_pcd(xyz1)
                reg = open3d.registration.registration_icp(pcd0, pcd1, 0.2, np.eye(4),
                                                        open3d.registration.TransformationEstimationPointToPoint(),
                                                        open3d.registration.ICPConvergenceCriteria(max_iteration=200))
                pcd0.transform(reg.transformation)
                M2 = M @ reg.transformation
                np.save(filename, M2)
            else:
                M2 = np.load(filename)
            self.kitti_icp_cache[key] = M2
        else:
            M2 = self.kitti_icp_cache[key]

        # refined pose is denoted as tsfm
        tsfm = M2
        rot = tsfm[:3,:3]
        trans = tsfm[:3,3][:,None]

        # add data augmentation
        src_pcd_input = copy.deepcopy(xyz0)
        tgt_pcd_input = copy.deepcopy(xyz1)
        if(self.data_augmentation):
            # add gaussian noise
            src_pcd_input += (np.random.rand(src_pcd_input.shape[0],3) - 0.5) * self.augment_noise
            tgt_pcd_input += (np.random.rand(tgt_pcd_input.shape[0],3) - 0.5) * self.augment_noise

            # rotate the point cloud
            euler_ab=np.random.rand(3)*np.pi*2 # anglez, angley, anglex
            rot_ab= Rotation.from_euler('zyx', euler_ab).as_matrix()
            if(np.random.rand(1)[0]>0.5):
                src_pcd_input = np.dot(rot_ab, src_pcd_input.T).T
                rot=np.matmul(rot,rot_ab.T)
            else:
                tgt_pcd_input = np.dot(rot_ab, tgt_pcd_input.T).T
                rot=np.matmul(rot_ab,rot)
                trans=np.matmul(rot_ab,trans)
            
            # scale the pcd
            scale = self.augment_scale_min + (self.augment_scale_max - self.augment_scale_min) * random.random()
            src_pcd_input = src_pcd_input * scale
            tgt_pcd_input = tgt_pcd_input * scale
            trans = scale * trans

        else:
            scale = 1

        # voxel down-sample the point clouds here
        _, sel_src = ME.utils.sparse_quantize(np.ascontiguousarray(src_pcd_input) / self.voxel_size, return_index=True)
        _, sel_tgt = ME.utils.sparse_quantize(np.ascontiguousarray(tgt_pcd_input) / self.voxel_size, return_index=True)

        # get correspondence
        tsfm = to_tsfm(rot, trans)
        src_xyz, tgt_xyz = src_pcd_input[sel_src], tgt_pcd_input[sel_tgt] # raw point clouds
        matching_inds = get_correspondences(to_o3d_pcd(src_xyz), to_o3d_pcd(tgt_xyz), tsfm, self.search_voxel_size * scale)
        if(matching_inds.size(0) < self.max_corr and self.split == 'train'):
            return self.__getitem__(np.random.choice(len(self.files),1)[0])

        # get voxelized coordinates
        src_coords, tgt_coords = np.floor(src_xyz / self.voxel_size), np.floor(tgt_xyz / self.voxel_size)

        # get feats
        src_feats = np.ones((src_coords.shape[0],1),dtype=np.float32)
        tgt_feats = np.ones((tgt_coords.shape[0],1),dtype=np.float32)

        src_xyz, tgt_xyz = to_tensor(src_xyz).float(), to_tensor(tgt_xyz).float()
        rot, trans = to_tensor(rot), to_tensor(trans)


        return src_xyz, tgt_xyz, src_coords, tgt_coords, src_feats, tgt_feats, matching_inds, rot, trans, scale


    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array([
                7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,
                -9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02
            ]).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext='.txt', return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + '/poses/%02d.txt' % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + '/sequences/%02d/velodyne/%06d.bin' % (drive, t)
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return (np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(
            np.linalg.inv(T1), T0).T)
