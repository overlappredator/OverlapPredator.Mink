# -*- coding: future_fstrings -*-
import torch
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
from models.common import get_norm
from models.gcn import GCN
import torch.nn.functional as F

from models.residual_block import get_block
import torch.nn as nn


class ResUNet2(ME.MinkowskiNetwork):
	NORM_TYPE = None
	BLOCK_NORM_TYPE = 'BN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 32, 64, 64, 128]

	# To use the model, must call initialize_coords before forward pass.
	# Once data is processed, call clear to reset the model before calling initialize_coords
	def __init__(self,config,D=3):
		ME.MinkowskiNetwork.__init__(self, D)
		NORM_TYPE = self.NORM_TYPE
		BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
		CHANNELS = self.CHANNELS
		TR_CHANNELS = self.TR_CHANNELS
		bn_momentum = config.bn_momentum
		self.normalize_feature = config.normalize_feature
		self.voxel_size = config.voxel_size

		self.conv1 = ME.MinkowskiConvolution(
			in_channels=config.in_feats_dim,
			out_channels=CHANNELS[1],
			kernel_size=config.conv1_kernel_size,
			stride=1,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, D=D)

		self.block1 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[1], CHANNELS[1], bn_momentum=bn_momentum, D=D)

		self.conv2 = ME.MinkowskiConvolution(
			in_channels=CHANNELS[1],
			out_channels=CHANNELS[2],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, D=D)

		self.block2 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[2], CHANNELS[2], bn_momentum=bn_momentum, D=D)

		self.conv3 = ME.MinkowskiConvolution(
			in_channels=CHANNELS[2],
			out_channels=CHANNELS[3],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, D=D)

		self.block3 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[3], CHANNELS[3], bn_momentum=bn_momentum, D=D)

		self.conv4 = ME.MinkowskiConvolution(
			in_channels=CHANNELS[3],
			out_channels=CHANNELS[4],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, D=D)

		self.block4 = get_block(
			BLOCK_NORM_TYPE, CHANNELS[4], CHANNELS[4], bn_momentum=bn_momentum, D=D)

		# adapt input tensor here
		self.conv4_tr = ME.MinkowskiConvolutionTranspose(
			in_channels = config.gnn_feats_dim + 2,
			out_channels=TR_CHANNELS[4],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm4_tr = get_norm(NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

		self.block4_tr = get_block(
			BLOCK_NORM_TYPE, TR_CHANNELS[4], TR_CHANNELS[4], bn_momentum=bn_momentum, D=D)

		self.conv3_tr = ME.MinkowskiConvolutionTranspose(
			in_channels=CHANNELS[3] + TR_CHANNELS[4],
			out_channels=TR_CHANNELS[3],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm3_tr = get_norm(NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

		self.block3_tr = get_block(
			BLOCK_NORM_TYPE, TR_CHANNELS[3], TR_CHANNELS[3], bn_momentum=bn_momentum, D=D)

		self.conv2_tr = ME.MinkowskiConvolutionTranspose(
			in_channels=CHANNELS[2] + TR_CHANNELS[3],
			out_channels=TR_CHANNELS[2],
			kernel_size=3,
			stride=2,
			dilation=1,
			bias=False,
			dimension=D)
		self.norm2_tr = get_norm(NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

		self.block2_tr = get_block(
			BLOCK_NORM_TYPE, TR_CHANNELS[2], TR_CHANNELS[2], bn_momentum=bn_momentum, D=D)

		self.conv1_tr = ME.MinkowskiConvolution(
			in_channels=CHANNELS[1] + TR_CHANNELS[2],
			out_channels=TR_CHANNELS[1],
			kernel_size=1,
			stride=1,
			dilation=1,
			bias=False,
			dimension=D)

		self.final = ME.MinkowskiConvolution(
			in_channels=TR_CHANNELS[1],
			out_channels=config.out_feats_dim + 2,
			kernel_size=1,
			stride=1,
			dilation=1,
			bias=True,
			dimension=D)


		#############
		# Overlap attention module
		self.epsilon = torch.nn.Parameter(torch.tensor(-5.0))
		self.bottle = nn.Conv1d(CHANNELS[4], config.gnn_feats_dim,kernel_size=1,bias=True)
		self.gnn = GCN(config.num_head,config.gnn_feats_dim, config.dgcnn_k, config.nets)
		self.proj_gnn = nn.Conv1d(config.gnn_feats_dim,config.gnn_feats_dim,kernel_size=1, bias=True)
		self.proj_score = nn.Conv1d(config.gnn_feats_dim,1,kernel_size=1,bias=True)
		



	def forward(self, stensor_src, stensor_tgt):
		################################
		# encode src
		src_s1 = self.conv1(stensor_src)
		src_s1 = self.norm1(src_s1)
		src_s1 = self.block1(src_s1)
		src = MEF.relu(src_s1)

		src_s2 = self.conv2(src)
		src_s2 = self.norm2(src_s2)
		src_s2 = self.block2(src_s2)
		src = MEF.relu(src_s2)

		src_s4 = self.conv3(src)
		src_s4 = self.norm3(src_s4)
		src_s4 = self.block3(src_s4)
		src = MEF.relu(src_s4)

		src_s8 = self.conv4(src)
		src_s8 = self.norm4(src_s8)
		src_s8 = self.block4(src_s8)
		src = MEF.relu(src_s8)


		################################
		# encode tgt
		tgt_s1 = self.conv1(stensor_tgt)
		tgt_s1 = self.norm1(tgt_s1)
		tgt_s1 = self.block1(tgt_s1)
		tgt = MEF.relu(tgt_s1)

		tgt_s2 = self.conv2(tgt)
		tgt_s2 = self.norm2(tgt_s2)
		tgt_s2 = self.block2(tgt_s2)
		tgt = MEF.relu(tgt_s2)

		tgt_s4 = self.conv3(tgt)
		tgt_s4 = self.norm3(tgt_s4)
		tgt_s4 = self.block3(tgt_s4)
		tgt = MEF.relu(tgt_s4)

		tgt_s8 = self.conv4(tgt)
		tgt_s8 = self.norm4(tgt_s8)
		tgt_s8 = self.block4(tgt_s8)
		tgt = MEF.relu(tgt_s8)


		################################
		# overlap attention module
		# empirically, when batch_size = 1, out.C[:,1:] == out.coordinates_at(0)		
		src_feats = src.F.transpose(0,1)[None,:]  #[1, C, N]
		tgt_feats = tgt.F.transpose(0,1)[None,:]  #[1, C, N]
		src_pcd, tgt_pcd = src.C[:,1:] * self.voxel_size, tgt.C[:,1:] * self.voxel_size

		# 1. project the bottleneck feature
		src_feats, tgt_feats = self.bottle(src_feats), self.bottle(tgt_feats)

		# 2. apply GNN to communicate the features and get overlap scores
		src_feats, tgt_feats= self.gnn(src_pcd.transpose(0,1)[None,:], tgt_pcd.transpose(0,1)[None,:],src_feats, tgt_feats)

		src_feats, src_scores = self.proj_gnn(src_feats), self.proj_score(src_feats)[0].transpose(0,1)
		tgt_feats, tgt_scores = self.proj_gnn(tgt_feats), self.proj_score(tgt_feats)[0].transpose(0,1)
		

		# 3. get cross-overlap scores
		src_feats_norm = F.normalize(src_feats, p=2, dim=1)[0].transpose(0,1)
		tgt_feats_norm = F.normalize(tgt_feats, p=2, dim=1)[0].transpose(0,1)
		inner_products = torch.matmul(src_feats_norm, tgt_feats_norm.transpose(0,1))
		temperature = torch.exp(self.epsilon) + 0.03
		src_scores_x = torch.matmul(F.softmax(inner_products / temperature ,dim=1) ,tgt_scores)
		tgt_scores_x = torch.matmul(F.softmax(inner_products.transpose(0,1) / temperature,dim=1),src_scores)

		# 4. update sparse tensor
		src_feats = torch.cat([src_feats[0].transpose(0,1), src_scores, src_scores_x], dim=1)
		tgt_feats = torch.cat([tgt_feats[0].transpose(0,1), tgt_scores, tgt_scores_x], dim=1)
		src = ME.SparseTensor(src_feats, 
			coordinate_map_key=src.coordinate_map_key,
			coordinate_manager=src.coordinate_manager)

		tgt = ME.SparseTensor(tgt_feats,
			coordinate_map_key=tgt.coordinate_map_key,
			coordinate_manager=tgt.coordinate_manager)


		################################
		# decoder src
		src = self.conv4_tr(src)
		src = self.norm4_tr(src)
		src = self.block4_tr(src)
		src_s4_tr = MEF.relu(src)

		src = ME.cat(src_s4_tr, src_s4)

		src = self.conv3_tr(src)
		src = self.norm3_tr(src)
		src = self.block3_tr(src)
		src_s2_tr = MEF.relu(src)

		src = ME.cat(src_s2_tr, src_s2)

		src = self.conv2_tr(src)
		src = self.norm2_tr(src)
		src = self.block2_tr(src)
		src_s1_tr = MEF.relu(src)

		src = ME.cat(src_s1_tr, src_s1)
		src = self.conv1_tr(src)
		src = MEF.relu(src)
		src = self.final(src)

		################################
		# decoder tgt
		tgt = self.conv4_tr(tgt)
		tgt = self.norm4_tr(tgt)
		tgt = self.block4_tr(tgt)
		tgt_s4_tr = MEF.relu(tgt)

		tgt = ME.cat(tgt_s4_tr, tgt_s4)

		tgt = self.conv3_tr(tgt)
		tgt = self.norm3_tr(tgt)
		tgt = self.block3_tr(tgt)
		tgt_s2_tr = MEF.relu(tgt)

		tgt = ME.cat(tgt_s2_tr, tgt_s2)

		tgt = self.conv2_tr(tgt)
		tgt = self.norm2_tr(tgt)
		tgt = self.block2_tr(tgt)
		tgt_s1_tr = MEF.relu(tgt)

		tgt = ME.cat(tgt_s1_tr, tgt_s1)
		tgt = self.conv1_tr(tgt)
		tgt = MEF.relu(tgt)
		tgt = self.final(tgt)

		################################
		# output features and scores
		sigmoid = nn.Sigmoid()
		src_feats, src_overlap, src_saliency = src.F[:,:-2], src.F[:,-2], src.F[:,-1]
		tgt_feats, tgt_overlap, tgt_saliency = tgt.F[:,:-2], tgt.F[:,-2], tgt.F[:,-1]

		src_overlap= torch.clamp(sigmoid(src_overlap.view(-1)),min=0,max=1)
		src_saliency = torch.clamp(sigmoid(src_saliency.view(-1)),min=0,max=1)
		tgt_overlap = torch.clamp(sigmoid(tgt_overlap.view(-1)),min=0,max=1)
		tgt_saliency = torch.clamp(sigmoid(tgt_saliency.view(-1)),min=0,max=1)

		src_feats = F.normalize(src_feats, p=2, dim=1)
		tgt_feats = F.normalize(tgt_feats, p=2, dim=1)

		scores_overlap = torch.cat([src_overlap, tgt_overlap], dim=0)
		scores_saliency = torch.cat([src_saliency, tgt_saliency], dim=0)

		return src_feats,  tgt_feats, scores_overlap, scores_saliency



class ResUNetBN2(ResUNet2):
  	NORM_TYPE = 'BN'


class ResUNetBN2B(ResUNet2):
	NORM_TYPE = 'BN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 64, 64, 64, 64]


class ResUNetBN2C(ResUNet2):
	NORM_TYPE = 'IN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 64, 64, 64, 128]
	BLOCK_NORM_TYPE = 'IN'

	# CHANNELS = [None, 64, 128, 256, 512]
	# TR_CHANNELS = [None, 64, 128, 128, 256]


class ResUNetBN2D(ResUNet2):
	NORM_TYPE = 'BN'
	CHANNELS = [None, 32, 64, 128, 256]
	TR_CHANNELS = [None, 64, 64, 128, 128]


class ResUNetBN2E(ResUNet2):
	NORM_TYPE = 'BN'
	CHANNELS = [None, 128, 128, 128, 256]
	TR_CHANNELS = [None, 64, 128, 128, 128]


class ResUNetIN2(ResUNet2):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2B(ResUNetBN2B):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2C(ResUNetBN2C):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2D(ResUNetBN2D):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'


class ResUNetIN2E(ResUNetBN2E):
	NORM_TYPE = 'BN'
	BLOCK_NORM_TYPE = 'IN'
