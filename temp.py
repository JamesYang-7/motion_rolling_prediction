import numpy as np
from glob import glob
from tqdm import tqdm
import torch

def main():
    data_path = 'dataset/BioMotionLab_NTroje/rub001/0000_treadmill_norm_stageii.npz'
    smpl_h_path = '/mnt/e/Data/AMASS/smplh/BioMotionLab_NTroje/rub005/0000_treadmill_norm_poses.npz'
    data = np.load(data_path, allow_pickle=True)
    print(data.files)
    print(data['gender'].item().upper())

def print_mean_std():
    mean = torch.load('datasets_processed/amass_p1_gendered_shaped/new_format_data/amass_p1_mean.pt')
    std = torch.load('datasets_processed/amass_p1_gendered_shaped/new_format_data/amass_p1_std.pt')
    mean1 = torch.load('/home/yy/Code/HmdBodyRecon/dataset/AMASS/amass_mean.pt')
    print('mean:', mean.shape)
    print('std:', std.shape)
    print('mean1:', mean1.shape)
    print('mean diff:', torch.abs(mean - mean1).max())

if __name__ == "__main__":
    # main()
    # print_mean_std()
    a = torch.load('/home/yy/Code/motion_rolling_prediction/datasets_processed/gorp/new_format_data/GORP/train/1.pt', weights_only=False)
    print(a.keys())
    for key in a.keys():
        print(key, a[key].shape)