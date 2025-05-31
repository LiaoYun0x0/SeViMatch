import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from modules.training.losses import *
from torchvision.ops import DeformConv2d
from sklearn.cluster import KMeans
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BasicLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
		super().__init__()
		self.layer = nn.Sequential(
									  nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
									  nn.BatchNorm2d(out_channels, affine=False),
									  nn.ReLU(inplace = True),
									)	
	def forward(self, x):
		return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)  

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.upsample(x) 
        return x

class MLP_FFN(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
  
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
   
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class VisualTokenAttention(nn.Module):
	def __init__(self, num_tokens, token_dim, norm_layer=nn.LayerNorm, mlp_ratio=4.,act_layer=nn.GELU, 
              drop_ratio=0., drop_path_ratio=0.,):
		super(VisualTokenAttention, self).__init__()
		self.num_tokens = num_tokens
		self.token_dim = token_dim
		self.norm1 = norm_layer(token_dim)	

		self.attention = nn.MultiheadAttention(embed_dim=token_dim, num_heads=8, batch_first=True)
		self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
		self.norm2 = norm_layer(token_dim)
		mlp_hidden_dim = int(token_dim * mlp_ratio)
		self.mlp = MLP_FFN(in_features=token_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)	
	def forward(self, visual_tokens, feature_map):
		B, C, H, W = feature_map.shape
		feature_map = feature_map.permute(0, 2, 3, 1).to(device) 
		flattened_feature_map = feature_map.view(B, H * W, C)  
		tokens = visual_tokens 
		
		for _ in range(2):
			attn_output, _ = self.attention(query=self.norm1(tokens), 
			                                key=self.norm1(flattened_feature_map), 
			                                value=self.norm1(flattened_feature_map))
			tokens = tokens + self.drop_path(attn_output)  
			tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))  
		return tokens 


class FeatureMapUpdateAttention(nn.Module):
	def __init__(self, token_dim, norm_layer=nn.LayerNorm, mlp_ratio=4.,act_layer=nn.GELU, 
              drop_ratio=0., drop_path_ratio=0.,):
		super(FeatureMapUpdateAttention, self).__init__()
		self.token_dim = token_dim
		self.norm1 = norm_layer(token_dim)
		self.attention = nn.MultiheadAttention(embed_dim=token_dim, num_heads=8, batch_first=True)
		self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
		self.norm2 = norm_layer(token_dim)
		mlp_hidden_dim = int(token_dim * mlp_ratio)
		self.mlp = MLP_FFN(in_features=token_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

	def forward(self, new_tokens, feature_map):
		B, C, H, W = feature_map.shape
		feature_map = feature_map.permute(0, 2, 3, 1).to(device)  
		flattened_feature_map = feature_map.view(B, H * W, C)  
            	
		for _ in range(2):
			attn_output, _ = self.attention(query=self.norm1(flattened_feature_map), 
        	                                  		key=self.norm1(new_tokens), 
        	                                        value=self.norm1(new_tokens))
			updated_feature_map = flattened_feature_map + self.drop_path(attn_output)  
			updated_feature_map = updated_feature_map + self.drop_path(self.mlp(self.norm2(updated_feature_map)))  
		updated_feature_map = updated_feature_map.view(B, H, W, C).permute(0, 3, 1, 2)  
		return updated_feature_map
    
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)  

    def forward(self, query, key, value):
        B, N, C = query.shape  
        _, S, _ = key.shape  
        
        query = query.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  
        key = key.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)      
        value = value.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  


        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale  
        attn_weights = self.softmax(attn_weights)

  
        output = torch.matmul(attn_weights, value)  

        # 合并多头输出
        output = output.permute(0, 2, 1, 3).reshape(B, N, C) 
        output = self.proj(output) 

        return output
    
class Block(nn.Module):
	def __init__(self,
                 dim,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
		super(Block, self).__init__()
		self.norm1 = norm_layer(dim)
		self.attn = Attention(dim, num_heads=num_heads)
		# NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
		self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
		self.norm2 = norm_layer(dim)
		mlp_hidden_dim = int(dim * mlp_ratio)
		self.mlp = MLP_FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)	
	
	def forward(self, x, y, is_selfatt = False):
		for _ in range(4): 
			if is_selfatt:
				x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x)))
				x = x + self.drop_path(self.mlp(self.norm2(x)))
				y = y + self.drop_path(self.attn(self.norm1(y), self.norm1(y), self.norm1(y)))
				y = y + self.drop_path(self.mlp(self.norm2(y)))
			
			
			x1 = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y), self.norm1(y)))  
			x1 = x1 + self.drop_path(self.mlp(self.norm2(x1))) 
			y1 = y + self.drop_path(self.attn(self.norm1(y), self.norm1(x), self.norm1(x))) 
			y1 = y1 + self.drop_path(self.mlp(self.norm2(y1))) 
		return x1, y1

class MultiScaleKeypointNet(nn.Module):
	def __init__(self, in_channels):
		super(MultiScaleKeypointNet, self).__init__()
		self.skip1 = nn.Sequential(	 nn.AvgPool2d(4, stride = 4),
			  						 nn.Conv2d (3, 64, 1, stride = 1, padding=0) )

		self.bl1 = nn.Sequential(
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
									 ) #[B, 24, H/4, W/4]
        
		
		self.offset_conv1 = nn.Conv2d(in_channels, 3 * 3 * 2, kernel_size=3, stride=4, padding=1)
		self.deform_conv1 = DeformConv2d(in_channels, 64, kernel_size=3, stride=4, padding=1)

		
		self.offset_conv2 = nn.Conv2d(64, 3 * 3 * 2, kernel_size=3, stride=2, padding=1)
		self.deform_conv2 = DeformConv2d(64, 64, kernel_size=3, stride=2, padding=1)

		
		self.offset_conv3 = nn.Conv2d(64, 3 * 3 * 2, kernel_size=3, stride=2, padding=1)
		self.deform_conv3 = DeformConv2d(64, 256, kernel_size=3, stride=2, padding=1)

	def forward(self, x):
		
		offset1 = self.offset_conv1(x)
		feat_1_4 = self.bl1(self.deform_conv1(x, offset1) + self.skip1(x))

		offset2 = self.offset_conv2(feat_1_4)
		feat_1_8 = self.deform_conv2(feat_1_4, offset2)

		offset3 = self.offset_conv3(feat_1_8)
		feat_1_16 = self.deform_conv3(feat_1_8, offset3)

		return feat_1_4, feat_1_8, feat_1_16

class DetailEnhancer(nn.Module):
    def __init__(self, channels):
        super(DetailEnhancer, self).__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.spatial_attention(x)  
        return x * (1 + attention) 

class NormalEnhancer(nn.Module):
    def __init__(self, channels):
        super(NormalEnhancer, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention = self.channel_attention(x)  
        return x * (1 + attention)  

class BackgroundEnhancer(nn.Module):
    def __init__(self, channels):
        super(BackgroundEnhancer, self).__init__()
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.global_context(x)  
        return x * (1 - context) + context  

class SeViMatchModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.norm = nn.InstanceNorm2d(3)
  
		self.encoder = Encoder(in_channels = 64)
		self.decoder = Decoder(out_channels = 64)
		self.weighting_module1 = DetailEnhancer(64) 
		self.weighting_module2 = NormalEnhancer(64)  
		self.weighting_module3 = BackgroundEnhancer(256) 
		self.block = Block(dim = 64)
		self.visual_token_attention = VisualTokenAttention(num_tokens = 3, token_dim = 64)
		self.feature_map_update_attention = FeatureMapUpdateAttention(token_dim = 64)
		self.deformconv = MultiScaleKeypointNet(in_channels = 3)

		self.skip1 = nn.Sequential(	 nn.AvgPool2d(8, stride = 8),
			  						 nn.Conv2d (3, 64, 1, stride = 1, padding=0) )


		self.bl2 = nn.Sequential(
										BasicLayer(64, 64, stride=1),
										BasicLayer(64, 64, stride=1),
									 ) #[B, 24, H/4, W/4]
  
		self.block_fusion =  nn.Sequential(
										BasicLayer(384, 64, stride=1),
										BasicLayer(64, 64, stride=1),
										nn.Conv2d (64, 64, 1, padding=0)
									 )

		self.heatmap_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 1, 1),
										nn.Sigmoid()
									)

		self.keypoint_head = nn.Sequential(
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										BasicLayer(64, 64, 1, padding=0),
										nn.Conv2d (64, 65, 1),
									)

		self.fine_matcher =  nn.Sequential(
											nn.Linear(128, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 512),
											nn.BatchNorm1d(512, affine=False),
									  		nn.ReLU(inplace = True),
											nn.Linear(512, 64),
										)
	
	def forward(self, x, y):
	
		with torch.no_grad():
			x = self.norm(x)
			y = self.norm(y)

		
		feature1A, feature2A, feature3A = self.deformconv(x)
		feature1B, feature2B, feature3B = self.deformconv(y)

		
		kpA = self.keypoint_head(feature2A)
		kpB = self.keypoint_head(feature2B)

		
		descriptor_downA = self.encoder(feature2A) 
		descriptor_downB = self.encoder(feature2B) 
		B1, _, H1, W1 = descriptor_downA.shape
		B2, _, H2, W2 = descriptor_downB.shape
		descriptor_downA = descriptor_downA.permute(0, 2, 3, 1).view(B1, H1*W1, 64)
		descriptor_downB = descriptor_downB.permute(0, 2, 3, 1).view(B2, H2*W2, 64)
		tokens_A , tokens_B = self.block(descriptor_downA, descriptor_downB, is_selfatt = True)  
		descriptor_upA = self.decoder(tokens_A.permute(0, 2, 1).view(B1, 64, H1, W1))  
		descriptor_upB = self.decoder(tokens_B.permute(0, 2, 1).view(B2, 64, H2, W2))

	
		_feats1A = self.weighting_module1(feature1A)  
		feats2A = self.weighting_module2(feature2A)
		_feats3A = self.weighting_module3(feature3A)
		feats1A = F.interpolate(_feats1A, (feats2A.shape[-2], feats2A.shape[-1]), mode='bilinear')
		feats3A = F.interpolate(_feats3A, (feats2A.shape[-2], feats2A.shape[-1]), mode='bilinear')
		featsA = self.block_fusion(torch.cat((feats1A, feats2A, feats3A), dim=1)) 
		featsA = self.bl2(featsA + self.skip1(x)) 
		

		
		feats1B = self.weighting_module1(feature1B)
		feats2B = self.weighting_module2(feature2B)
		feats3B = self.weighting_module3(feature3B)
		feats1B = F.interpolate(feats1B, (feats2B.shape[-2], feats2B.shape[-1]), mode='bilinear')
		feats3B = F.interpolate(feats3B, (feats2B.shape[-2], feats2B.shape[-1]), mode='bilinear')
		featsB = self.block_fusion(torch.cat((feats1B, feats2B, feats3B), dim=1))
		featsB = self.bl2(featsB + self.skip1(y))

	
		visual_tokens = torch.cat((tokens_A, tokens_B), dim=1) 
		b, n, c = visual_tokens.shape
		visual_tokens_np = visual_tokens.detach().cpu().numpy().reshape(-1, c)
		kmeans_results = []
		for i in range(b):
			kmeans = KMeans(n_clusters=10, n_init='auto').fit(visual_tokens_np[i * n:(i + 1) * n]) 
			kmeans_results.append(kmeans.cluster_centers_)
		visual_tokens_clustered = torch.tensor(np.stack(kmeans_results), device=device)
		visual_tokenA = self.visual_token_attention(visual_tokens_clustered, featsA)  
		visual_tokenB = self.visual_token_attention(visual_tokens_clustered, featsB)
		visual_tokenA, visual_tokenB = self.block(visual_tokenA, visual_tokenB)


		descriptorA = self.feature_map_update_attention(visual_tokenA, featsA) 
		descriptorB = self.feature_map_update_attention(visual_tokenB, featsB)


		#heads
		heatmapA = self.heatmap_head(descriptorA) 
		heatmapB = self.heatmap_head(descriptorB)

		
		return descriptorA, kpA, heatmapA, descriptorB, kpB, heatmapB, feature2A, feature2B, descriptor_upA, descriptor_upB, visual_tokenA, visual_tokenB

