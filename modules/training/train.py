import argparse
import os
import time
import sys
from sklearn.cluster import KMeans

def parse_arguments():
    parser = argparse.ArgumentParser(description="SeViMatch training script.")

    parser.add_argument('--megadepth_root_path', type=str, default='/first_disk/MegaDepth_v1',
                        help='Path to the MegaDepth dataset root directory.')
    parser.add_argument('--synthetic_root_path', type=str, default='assets/coco_20k',
                        help='Path to the synthetic dataset root directory.')
    parser.add_argument('--ckpt_save_path', type=str, default='ckpts',
                        help='Path to save the checkpoints.')
    parser.add_argument('--training_type', type=str, default='SeViMatch_default',
                        choices=['SeViMatch_default', 'SeViMatch_synthetic', 'SeViMatch_megadepth'],
                        help='Training scheme. SeViMatch_default uses both megadepth & synthetic warps.')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for training. Default is 10.')
    parser.add_argument('--n_steps', type=int, default=320000,
                        help='Number of training steps. Default is 160000.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate. Default is 0.0003.')
    parser.add_argument('--gamma_steplr', type=float, default=0.5,
                        help='Gamma value for StepLR scheduler. Default is 0.5.')
    parser.add_argument('--training_res', type=lambda s: tuple(map(int, s.split(','))),
                        default=(640, 640), help='Training resolution as width,height. Default is (640, 640).')
    parser.add_argument('--device_num', type=str, default='0',
                        help='Device number to use for training. Default is "0".')
    parser.add_argument('--dry_run', action='store_true',
                        help='If set, perform a dry run training with a mini-batch for sanity check.')
    parser.add_argument('--save_ckpt_every', type=int, default=10000,
                        help='Save checkpoints every N steps. Default is 500.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    return args

args = parse_arguments()

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np

# 获取父级目录（modules/ 的上一级目录，即 project/）
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(parent_dir)

from modules.model import *
from modules.dataset.augmentation import *
from modules.training.utils import *
from modules.training.losses import *

from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.dataset.megadepth import megadepth_warper
from torch.utils.data import Dataset, DataLoader


class Trainer():

    def __init__(self, megadepth_root_path, 
                       synthetic_root_path, 
                       ckpt_save_path, 
                       model_name = 'SeViMatch_default',
                       batch_size = 4, n_steps = 160_000, lr= 3e-4, gamma_steplr=0.5, 
                       training_res = (640, 640), device_num="0", dry_run = False,
                       save_ckpt_every = 500):

        self.dev = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = SeViMatchModel().to(self.dev)

        #Setup optimizer 
        self.batch_size = batch_size
        self.steps = n_steps
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()) , lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=30_000, gamma=gamma_steplr)

        ##################### Synthetic COCO INIT ##########################
        if model_name in ('SeViMatch_default', 'SeViMatch_synthetic'):
            self.augmentor = AugmentationPipe(
                                        img_dir = synthetic_root_path,
                                        device = self.dev, load_dataset = True,
                                        batch_size = int(self.batch_size * 0.4 if model_name=='SeViMatch_default' else batch_size),
                                        out_resolution = training_res, 
                                        warp_resolution = training_res,
                                        sides_crop = 0.1,
                                        max_num_imgs = 3_000,
                                        num_test_imgs = 5,
                                        photometric = True,
                                        geometric = True,
                                        reload_step = 4_000
                                        )
        else:
            self.augmentor = None
        ##################### Synthetic COCO END #######################


        ##################### MEGADEPTH INIT ##########################
        if model_name in ('SeViMatch_default', 'SeViMatch_megadepth'):
            TRAIN_BASE_PATH = f"{megadepth_root_path}/megadepth_indices"
            TRAINVAL_DATA_SOURCE = f"{megadepth_root_path}/phoenix/S6/zl548/MegaDepth_v1"

            TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"

            npz_paths = glob.glob(TRAIN_NPZ_ROOT + '/*.npz')[:]
            data = torch.utils.data.ConcatDataset( [MegaDepthDataset(root_dir = TRAINVAL_DATA_SOURCE,
                            npz_path = path) for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")] )

            self.data_loader = DataLoader(data, 
                                          batch_size=int(self.batch_size * 0.6 if model_name=='SeViMatch_default' else batch_size),
                                          shuffle=True)
            self.data_iter = iter(self.data_loader)

        else:
            self.data_iter = None
        ##################### MEGADEPTH INIT END #######################

        os.makedirs(ckpt_save_path, exist_ok=True)
        os.makedirs(ckpt_save_path + '/logdir', exist_ok=True)

        self.dry_run = dry_run
        self.save_ckpt_every = save_ckpt_every
        self.ckpt_save_path = ckpt_save_path
        self.writer = SummaryWriter(ckpt_save_path + f'/logdir/{model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S"))
        self.model_name = model_name
    
        self.best_loss = 4.5  # 初始化最优损失为 10
        self.best_auc = 0.80
        self.best_weights = None  # 用于保存最优权重路径

    def train(self):

        self.net.train()


        difficulty = 0.10

        p1s, p2s, H1, H2 = None, None, None, None
        d = None

        
        if self.augmentor is not None:
            p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)
        
        
        if self.data_iter is not None:
            d = next(self.data_iter)

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                if not self.dry_run:
                    if self.data_iter is not None:
                        try:
                           
                            d = next(self.data_iter)

                        except StopIteration:
                            print("End of DATASET!")
                            # If StopIteration is raised, create a new iterator.
                            self.data_iter = iter(self.data_loader)
                            d = next(self.data_iter)

                    if self.augmentor is not None:
                        #Grab synthetic data
                        p1s, p2s, H1, H2 = make_batch(self.augmentor, difficulty)

                if d is not None:
                    for k in d.keys():
                        if isinstance(d[k], torch.Tensor):
                            d[k] = d[k].to(self.dev)
                
                    p1, p2 = d['image0'], d['image1']
                    positives_md_coarse = megadepth_warper.spvs_coarse(d, 8)

                if self.augmentor is not None:
                    h_coarse, w_coarse = p1s[0].shape[-2] // 8, p1s[0].shape[-1] // 8
                    _ , positives_s_coarse = get_corresponding_pts(p1s, p2s, H1, H2, self.augmentor, h_coarse, w_coarse)

                
                with torch.inference_mode():
                    if self.model_name in ('sevimatch_default'):
                        p1 = torch.cat([p1s, p1], dim=0)
                        p2 = torch.cat([p2s, p2], dim=0)
                        positives_c = positives_s_coarse + positives_md_coarse
                    elif self.model_name in ('sevimatch_synthetic'):
                        p1 = p1s ; p2 = p2s
                        positives_c = positives_s_coarse
                    else:
                        positives_c = positives_md_coarse

                is_corrupted = False
                for p in positives_c:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    continue

                feats1, kpts1_8, hmap1,feats2, kpts2_8, hmap2, featureA, featureB, featureA_up, featureB_up,_,_ = self.net(p1, p2)
                
             
                loss_items = []
                for b in range(len(positives_c)):
                    
                    pts1, pts2 = positives_c[b][:, :2], positives_c[b][:, 2:]

                   
                    m1 = feats1[b, :, pts1[:,1].long(), pts1[:,0].long()].permute(1,0)
                    m2 = feats2[b, :, pts2[:,1].long(), pts2[:,0].long()].permute(1,0)
                    
                    
                    h1 = hmap1[b, 0, pts1[:,1].long(), pts1[:,0].long()]
                    h2 = hmap2[b, 0, pts2[:,1].long(), pts2[:,0].long()]
                    
                   
                    coords1 = self.net.fine_matcher(torch.cat([m1, m2], dim=-1))

                    loss_rebuild1 = reconstruction_loss(featureA, featureA_up)
                    loss_rebuild2 = reconstruction_loss(featureB, featureB_up)
                    loss_rebuild = (loss_rebuild1 + loss_rebuild2)*20.0
                   
                    loss_ds, conf = dual_softmax_loss(m1, m2)
            
                  
                    loss_coords, acc_coords = coordinate_classification_loss(coords1, pts1, pts2, conf)

                    
                    loss_kp_pos1, acc_pos1 = alike_distill_loss(kpts1_8[b], p1[b], 8)
                    loss_kp_pos5, acc_pos5 = alike_distill_loss(kpts2_8[b], p2[b], 8)
                    loss_pos = (loss_kp_pos1 + loss_kp_pos5)*2.0

                    
                    loss_kp =  keypoint_loss(h1, conf) + keypoint_loss(h2, conf)
                    
                    loss_items.append(loss_rebuild.unsqueeze(0))
                    loss_items.append(loss_ds.unsqueeze(0))
                    loss_items.append(loss_coords.unsqueeze(0))
                    loss_items.append(loss_pos.unsqueeze(0))
                    loss_items.append(loss_kp.unsqueeze(0))
                    
                    if b == 0:
                        acc_coarse_0 = check_accuracy(m1, m2)

                
                acc_coarse = check_accuracy(m1, m2)

                nb_coarse = len(m1)
                loss = torch.cat(loss_items, -1).mean()
                loss_rebuild = loss_rebuild.item()
                loss_coarse = loss_ds.item()
                loss_coord = loss_coords.item()
                loss_pos = loss_pos.item()
                loss_l1 = loss_kp.item()

               
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                self.opt.step()
                self.opt.zero_grad()
                self.scheduler.step()

                if loss.item() < self.best_loss or acc_coarse_0 > self.best_auc or (i+1) % self.save_ckpt_every == 0:
                    if loss.item() < self.best_loss:
                        self.best_loss = loss.item()
                    if acc_coarse_0 > self.best_auc:
                        self.best_auc = acc_coarse_0
                    self.best_weights = f"{self.ckpt_save_path}/loss{loss.item():.3f}_auc{acc_coarse_0:.3f}_{i+1}.pth"
                    torch.save(self.net.state_dict(), self.best_weights)
                    print(f"保存最优权重: {self.best_weights}, 损失: {self.best_loss}")

                pbar.set_description( '总损失: {:.4f} acc_coarse {:.3f} loss_rebuild {:.3f} loss_ds {:.3f} loss_coord: {:.3f} loss_pos: {:.3f} loss_heat: {:.3f}'.format(                                                        
                                        loss.item(), acc_coarse, loss_rebuild, loss_ds, loss_coord, loss_pos, loss_l1) )
                pbar.update(1)

                # Log metrics
                self.writer.add_scalar('Loss/total', loss.item(), i)
                self.writer.add_scalar('Accuracy/coarse_synth', acc_coarse_0, i)
                self.writer.add_scalar('Accuracy/coarse_mdepth', acc_coarse, i)
                self.writer.add_scalar('Loss/coarse', loss_coarse, i)
                self.writer.add_scalar('Loss/fine', loss_coord, i)
                self.writer.add_scalar('Loss/reliability', loss_l1, i)
                self.writer.add_scalar('Loss/keypoint', loss_kp, i)
                self.writer.add_scalar('Count/matches_coarse', nb_coarse, i)



if __name__ == '__main__':

    trainer = Trainer(
        megadepth_root_path=args.megadepth_root_path, 
        synthetic_root_path=args.synthetic_root_path, 
        ckpt_save_path=args.ckpt_save_path,
        model_name=args.training_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        lr=args.lr,
        gamma_steplr=args.gamma_steplr,
        training_res=args.training_res,
        device_num=args.device_num,
        dry_run=args.dry_run,
        save_ckpt_every=args.save_ckpt_every
    )

    #The most fun part
    trainer.train()
