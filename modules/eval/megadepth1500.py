import argparse, glob, sys, os, time
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import poselib
import json
import copy
import tqdm


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from modules.dataset.megadepth.megadepth import MegaDepthDataset
from modules.mymodel import SeViMatch
np.set_printoptions(suppress=True)

class MegaDepth1500(Dataset):
    """
        Streamlined MegaDepth-1500 dataloader. The camera poses & metadata are stored in a formatted json for facilitating 
        the download of the dataset and to keep the setup as simple as possible.
    """
    def __init__(self, json_file, root_dir):
        # Load the info & calibration from the JSON
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.root_dir = root_dir

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
            f"Dataset {self.root_dir} does not exist! \n \
              > If you didn't download the dataset, use the downloader tool: python3 -m modules.dataset.download -h")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])

        h1, w1 = data['size0_hw']
        h2, w2 = data['size1_hw']

        image0 = cv2.resize( cv2.imread(f"{self.root_dir}/{data['pair_names'][0]}"),
                             (w1, h1))

        image1 = cv2.resize( cv2.imread(f"{self.root_dir}/{data['pair_names'][1]}"),
                             (w2, h2))

        data['image0'] = torch.tensor(image0.astype(np.float32)/255).permute(2,0,1)
        data['image1'] = torch.tensor(image1.astype(np.float32)/255).permute(2,0,1)

        for k,v in data.items():
            if k not in ('dataset_name', 'scene_id', 'pair_id', 'pair_names', 'size0_hw', 'size1_hw', 'image0', 'image1'):
                data[k] = torch.tensor(np.array(v, dtype=np.float32))

        return data


################################# Metrics #####################################

def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    # angle error between 2 vectors
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)
    t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
    t_err = np.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if np.linalg.norm(t_gt) < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # handle numercial errors
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def intrinsics_to_camera(K):
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }

def estimate_pose_poselib(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    M, info = poselib.estimate_relative_pose(
        kpts0, kpts1,
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
        {"max_epipolar_error": thresh,
         "success_prob": conf,
         "min_iterations": 20,
         "max_iterations": 1_000},
    )

    R, t, inl = M.R, M.t, info["inliers"]
    inl = np.array(inl)
    ret = (R, t, inl)

    return ret, (kpts0, kpts1)


def tensor2bgr(t):
    return (t.cpu()[0].permute(1,2,0).numpy()*255).astype(np.uint8)


def compute_pose_error(pair):
    
    pixel_thr = 1.0 if 'ransac_thr' not in pair else pair['ransac_thr']
    conf = 0.99999
    pair.update({'R_err':  np.inf, 't_err': np.inf, 'inliers': []})

    pts0 = pair['pts0']
    pts1 = pair['pts1']
    K0 = pair['K0'].cpu().numpy()[0]
    K1 = pair['K1'].cpu().numpy()[0]
    T_0to1 = pair['T_0to1'].cpu().numpy()[0]

    ret, corrs = estimate_pose_poselib(pts0, pts1, K0, K1, pixel_thr, conf=conf)

    if ret is not None:
        R, t, inliers = ret

        t_err, R_err = relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0)

        pair['R_err'] = R_err
        pair['t_err'] = t_err


def error_auc(errors, thresholds=[5, 10, 20]):
   
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []

    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}

def compute_maa(pairs, thresholds=[5, 10, 20]):
    print("auc / mAcc on %d pairs" % (len(pairs)))
    errors = []

    for p in pairs:
        et = p['t_err']
        er = p['R_err']
        errors.append(max(et, er))

    d_err_auc = error_auc(errors)

    for k,v in d_err_auc.items():
        print(k, ': ', '%.1f'%(v*100))

    errors = np.array(errors)

    for t in thresholds:
        acc = (errors <= t).sum() / len(errors)
        print("mAcc@%d: %.1f "%(t, acc*100))
    

@torch.inference_mode()
def run_pose_benchmark(matcher_fn, loader, ransac_thr=2.5):
    pairs = []
    cnt = 0
    for d in tqdm.tqdm(loader):
        d_error = {}
        src_pts, dst_pts = matcher_fn(tensor2bgr(d['image0']), tensor2bgr(d['image1']))

        #delete images to avoid OOM, happens in low mem machines
        del d['image0']
        del d['image1']

        #rescale kpts
        src_pts = src_pts * d['scale0'].numpy()
        dst_pts = dst_pts * d['scale1'].numpy()
        d.update({"pts0":src_pts, "pts1": dst_pts,'ransac_thr': ransac_thr})
        compute_pose_error(d)
        pairs.append(d)
        cnt+=1

    compute_maa(pairs)

def parse_args():
    parser = argparse.ArgumentParser(description="Run pose benchmark with matcher")
    parser.add_argument('--dataset-dir', type=str, default='assets/test',
                        help="Path to MegaDepth dataset root")
    parser.add_argument('--matcher', type=str, default='mymodel', help="Matcher to use SeViMatch")
    parser.add_argument('--ransac-thr', type=float, default=2.5,
                        help="RANSAC threshold value in pixels (default: 2.5)")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    dataset = MegaDepth1500( json_file = './assets/megadepth_1500.json',
                             root_dir =  args.dataset_dir + "/megadepth_test_1500")

    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Running benchmark for mymodel..")
    

    weights_dir = 'ckpts'
    for filename in os.listdir(weights_dir):
        if filename.endswith('.pth'):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            weights_path = os.path.join(weights_dir, filename)
            print(f"Loading weights from {weights_path}")

            # 加载权重并运行基准测试
            sevimatch = SeViMatch(weights=weights_path, top_k = 10_000)
            run_pose_benchmark(matcher_fn=sevimatch.match_sevimatch, loader=loader, ransac_thr=args.ransac_thr)
