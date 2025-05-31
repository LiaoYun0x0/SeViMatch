
import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

import tqdm

from modules.model import *
from modules.interpolator import InterpolateSparse2d

class SeViMatch(nn.Module):

	def __init__(self,  weights=None, top_k = 4096, detection_threshold=0.05):
		super().__init__()
		self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.net = SeViMatchModel().to(self.dev).eval()
		self.top_k = top_k
		self.detection_threshold = detection_threshold

		if weights is not None:
			if isinstance(weights, str):
				self.net.load_state_dict(torch.load(weights, map_location=self.dev, weights_only=True))
			else:
				self.net.load_state_dict(weights)

		self.interpolator = InterpolateSparse2d('bicubic')

		#Try to import LightGlue from Kornia
		self.kornia_available = False
		self.lighterglue = None
		try:
			import kornia
			self.kornia_available=True
		except:
			pass


	@torch.inference_mode()
	def detectAndCompute(self, x, y, top_k = None, detection_threshold = None):
		
		if top_k is None: top_k = self.top_k
		if detection_threshold is None: detection_threshold = self.detection_threshold
		x, rh1, rw1 = self.preprocess_tensor(x)
		y, rh2, rw2 = self.preprocess_tensor(y)

		B1, _, _H1, _W1 = x.shape
		B2, _, _H2, _W2 = y.shape
		
		M1, K1_8, H1, M2, K2_8, H2, feature2A, feature2B,_,_, visual_tokenA, visual_tokenB= self.net(x, y)
		M1 = F.normalize(M1, dim=1)
		M2 = F.normalize(M2, dim=1)


		
		K1h = self.get_kpts_heatmap(K1_8, classnum = 8) 
		mkpts1 = self.NMS(K1h, threshold=detection_threshold, kernel_size=5) 

		
		_nearest = InterpolateSparse2d('nearest')
		_bilinear = InterpolateSparse2d('bilinear')

		
		scores1 = (_nearest(K1h, mkpts1, _H1, _W1) * _bilinear(H1, mkpts1, _H1, _W1)).squeeze(-1)
		scores1[torch.all(mkpts1 == 0, dim=-1)] = -1

		
		idxs1 = torch.argsort(-scores1, dim=-1)
		mkpts1_x  = torch.gather(mkpts1[...,0], -1, idxs1)[:, :top_k]
		mkpts1_y  = torch.gather(mkpts1[...,1], -1, idxs1)[:, :top_k]
		mkpts1 = torch.cat([mkpts1_x[...,None], mkpts1_y[...,None]], dim=-1)
		scores1 = torch.gather(scores1, -1, idxs1)[:, :top_k]

		
		feats1 = self.interpolator(M1, mkpts1, H = _H1, W = _W1)

		
		feats1 = F.normalize(feats1, dim=-1)

		
		mkpts1 = mkpts1 * torch.tensor([rw1,rh1], device=mkpts1.device).view(1, 1, -1)

		valid1 = scores1 > 0
		
	
		Kh2 = self.get_kpts_heatmap(K2_8, classnum = 8)
		
		mkpts2 = self.NMS(Kh2, threshold=detection_threshold, kernel_size=5) 
		

		
		_nearest = InterpolateSparse2d('nearest')
		_bilinear = InterpolateSparse2d('bilinear')

		
		scores2 = (_nearest(Kh2, mkpts2, _H2, _W2) * _bilinear(H2, mkpts2, _H2, _W2)).squeeze(-1)
		scores2[torch.all(mkpts2 == 0, dim=-1)] = -1

	
		idxs2 = torch.argsort(-scores2, dim=-1)
		mkpts2_x  = torch.gather(mkpts2[...,0], -1, idxs2)[:, :top_k]
		mkpts2_y  = torch.gather(mkpts2[...,1], -1, idxs2)[:, :top_k]
		mkpts2 = torch.cat([mkpts2_x[...,None], mkpts2_y[...,None]], dim=-1)
		scores2 = torch.gather(scores2, -1, idxs2)[:, :top_k]

		
		feats2 = self.interpolator(M2, mkpts2, H = _H2, W = _W2)

		
		feats2 = F.normalize(feats2, dim=-1)

		mkpts2 = mkpts2 * torch.tensor([rw2,rh2], device=mkpts2.device).view(1, 1, -1)

		valid2 = scores2 > 0
	

		dict1 = [  
				   {'keypoints': mkpts1[b][valid1[b]],
					'scores': scores1[b][valid1[b]],
					'descriptors': feats1[b][valid1[b]],
					'visual_token': visual_tokenA[b],
					'feature':M1[b],
					} for b in range(B1) 
			   ]

		dict2 = [  
				   {'keypoints': mkpts2[b][valid2[b]],
					'scores': scores2[b][valid2[b]],
					'descriptors': feats2[b][valid2[b]],
					'visual_token':visual_tokenB[b],
					'feature':M2[b]
					} for b in range(B2) 
			   ]

		return dict1[0], dict2[0]


	@torch.inference_mode()
	def match_sevimatch(self, img1, img2, top_k = None, min_cossim = -1):
	
		if top_k is None: top_k = self.top_k
		img1 = self.parse_input(img1)
		img2 = self.parse_input(img2)

		out1,out2= self.detectAndCompute(img1,img2, top_k=top_k)
		idxs0, idxs1 = self.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim )
		return out1['keypoints'][idxs0].cpu().numpy(), out2['keypoints'][idxs1].cpu().numpy()



	def preprocess_tensor(self, x):
		""" Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
		if isinstance(x, np.ndarray) and len(x.shape) == 3:
			x = torch.tensor(x).permute(2,0,1)[None]
		x = x.to(self.dev).float()

		H, W = x.shape[-2:]
		_H, _W = (H//32) * 32, (W//32) * 32
		rh, rw = H/_H, W/_W

		x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
		return x, rh, rw

	def get_kpts_heatmap(self, kpts, classnum, softmax_temp = 1.0):
		scores = F.softmax(kpts*softmax_temp, 1)[:, :classnum**2] #[1, 64, H/8, W/8]
		B, _, H, W = scores.shape
		heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, classnum, classnum) # [1, H/8, W/8, 8, 8]
		heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H*classnum, W*classnum) # [1, 1, H, W]
		return heatmap

	def NMS(self, x, threshold = 0.05, kernel_size = 5):
		B, _, H, W = x.shape
		pad=kernel_size//2
		local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
		pos = (x == local_max) & (x > threshold)
		pos_batched = [k.nonzero()[..., 1:].flip(-1) for k in pos]

		pad_val = max([len(x) for x in pos_batched])
		pos = torch.zeros((B, pad_val, 2), dtype=torch.long, device=x.device)

		#Pad kpts and build (B, N, 2) tensor
		for b in range(len(pos_batched)):
			pos[b, :len(pos_batched[b]), :] = pos_batched[b]

		return pos

	@torch.inference_mode()
	def batch_match(self, feats1, feats2, min_cossim = -1):
		B = len(feats1)
		cossim = torch.bmm(feats1, feats2.permute(0,2,1))
		match12 = torch.argmax(cossim, dim=-1)
		match21 = torch.argmax(cossim.permute(0,2,1), dim=-1)

		idx0 = torch.arange(len(match12[0]), device=match12.device)

		batched_matches = []

		for b in range(B):
			mutual = match21[b][match12[b]] == idx0

			if min_cossim > 0:
				cossim_max, _ = cossim[b].max(dim=1)
				good = cossim_max > min_cossim
				idx0_b = idx0[mutual & good]
				idx1_b = match12[b][mutual & good]
			else:
				idx0_b = idx0[mutual]
				idx1_b = match12[b][mutual]

			batched_matches.append((idx0_b, idx1_b))

		return batched_matches

	def subpix_softmax2d(self, heatmaps, temp = 3):
		N, H, W = heatmaps.shape
		heatmaps = torch.softmax(temp * heatmaps.view(-1, H*W), -1).view(-1, H, W)
		x, y = torch.meshgrid(torch.arange(W, device =  heatmaps.device ), torch.arange(H, device =  heatmaps.device ), indexing = 'xy')
		x = x - (W//2)
		y = y - (H//2)

		coords_x = (x[None, ...] * heatmaps)
		coords_y = (y[None, ...] * heatmaps)
		coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H*W, 2)
		coords = coords.sum(1)

		return coords

	def refine_matches(self, d0, d1, matches, batch_idx, fine_conf = 0.25):
		idx0, idx1 = matches[batch_idx]
		feats1 = d0['descriptors'][batch_idx][idx0]
		feats2 = d1['descriptors'][batch_idx][idx1]
		mkpts_0 = d0['keypoints'][batch_idx][idx0]
		mkpts_1 = d1['keypoints'][batch_idx][idx1]
		sc0 = d0['scales'][batch_idx][idx0]

		#Compute fine offsets
		offsets = self.net.fine_matcher(torch.cat([feats1, feats2],dim=-1))
		conf = F.softmax(offsets*3, dim=-1).max(dim=-1)[0]
		offsets = self.subpix_softmax2d(offsets.view(-1,8,8))

		mkpts_0 += offsets* (sc0[:,None]) #*0.9 #* (sc0[:,None])

		mask_good = conf > fine_conf
		mkpts_0 = mkpts_0[mask_good]
		mkpts_1 = mkpts_1[mask_good]

		return torch.cat([mkpts_0, mkpts_1], dim=-1)

	@torch.inference_mode()
	def match(self, feats1, feats2, min_cossim = 0.82):

		cossim = feats1 @ feats2.t()
		cossim_t = feats2 @ feats1.t()
		
		_, match12 = cossim.max(dim=1)
		_, match21 = cossim_t.max(dim=1)

		idx0 = torch.arange(len(match12), device=match12.device)
		mutual = match21[match12] == idx0

		if min_cossim > 0:
			cossim, _ = cossim.max(dim=1)
			good = cossim > min_cossim
			idx0 = idx0[mutual & good]
			idx1 = match12[mutual & good]
		else:
			idx0 = idx0[mutual]
			idx1 = match12[mutual]

		return idx0, idx1

	def create_xy(self, h, w, dev):
		y, x = torch.meshgrid(torch.arange(h, device = dev), 
								torch.arange(w, device = dev), indexing='ij')
		xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
		return xy


	def extract_dualscale(self, x, y, top_k, s1 = 0.6, s2 = 1.3):
		x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
		x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')
		y1 = F.interpolate(y, scale_factor=s1, align_corners=False, mode='bilinear')
		y2 = F.interpolate(y, scale_factor=s2, align_corners=False, mode='bilinear')

		B1, _, _, _ = x.shape
		B2, _, _, _ = y.shape

		mkpts1_1, feats1_1, mkpts2_1, feats2_1 = self.extractDense(x1, y1, int(top_k*0.20))
		mkpts1_2, feats1_2, mkpts2_2, feats2_2 = self.extractDense(x2, y2, int(top_k*0.80))

		mkpts1 = torch.cat([mkpts1_1/s1, mkpts1_2/s2], dim=1)
		sc1_1 = torch.ones(mkpts1_1.shape[:2], device=mkpts1_1.device) * (1/s1)
		sc1_2 = torch.ones(mkpts1_2.shape[:2], device=mkpts1_2.device) * (1/s2)
		sc1 = torch.cat([sc1_1, sc1_2],dim=1)
		feats1 = torch.cat([feats1_1, feats1_2], dim=1)

		mkpts2 = torch.cat([mkpts2_1/s1, mkpts2_2/s2], dim=1)
		sc2_1 = torch.ones(mkpts2_1.shape[:2], device=mkpts2_1.device) * (1/s1)
		sc2_2 = torch.ones(mkpts2_2.shape[:2], device=mkpts2_2.device) * (1/s2)
		sc2 = torch.cat([sc2_1, sc2_2],dim=1)
		feats2 = torch.cat([feats2_1, feats2_2], dim=1)

		return mkpts1, sc1, feats1, mkpts2, sc2, feats2

	def parse_input(self, x):
		if len(x.shape) == 3:
			x = x[None, ...]

		if isinstance(x, np.ndarray):
			x = torch.tensor(x).permute(0,3,1,2)/255

		return x
