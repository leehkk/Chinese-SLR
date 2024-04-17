import csv
import random

import cv2
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.video import swin3d_t, Swin3D_T_Weights,Swin3D_S_Weights,swin3d_s
import numpy as np
import torch
import os
from torchvision.transforms import transforms
import logging
from Swin.CSL_S import CSL
import torch.nn.functional as F
from fvcore.common.file_io import PathManager


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def swin_get_rgb(path):
    video = cv2.VideoCapture(path)
    video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = video_frames // 32
    index_list = frame_indices_tranform(video_frames, 32)
    indices = np.arange(0, video_frames, step)[:32]
    frames = []
    for i in index_list:
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if ret:
            frames.append(frame)
    frames = np.array(frames)
    frames_tensor = torch.from_numpy(frames)
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)
    weights = Swin3D_S_Weights.DEFAULT
    preprocess = weights.transforms()
    frames = preprocess(frames_tensor)
    # print(images.shape)
    return frames

def frame_indices_tranform(video_length, sample_duration):
    if video_length > sample_duration:
        random_start = random.randint(0, video_length - sample_duration)
        frame_indices = np.arange(random_start, random_start + sample_duration)
    else:
        frame_indices = np.arange(video_length)
        while frame_indices.shape[0] < sample_duration:
            frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
        frame_indices = frame_indices[:sample_duration]
    assert frame_indices.shape[0] == sample_duration
    return frame_indices

def center_crop(input_width, input_height, output_size):
    i = (input_width - output_size) // 2
    j = (input_height - output_size) // 2
    return i, j, i + output_size, j + output_size

def save_rgb_feature(save_path,csl_path,num_classes,weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_model = swin3d_s(weights=None)
    rgb_model.head = nn.Linear(768, num_classes)
    state_dict = torch.load(weight_path)
    rgb_model.load_state_dict(state_dict)
    rgb_model.head = nn.Sequential()
    print(rgb_model)
    rgb_model = rgb_model.to(device)
    with PathManager.open(csl_path, "r") as f:
        with open(save_path, "w", newline="") as rgb_file:
            rgb_writer = csv.writer(rgb_file, lineterminator='\n')
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                path, label = path_label.split(",")
                print(label)
                rgb_frames = swin_get_rgb(path).unsqueeze(0).to(device)
                rgb_pred = rgb_model(rgb_frames)
                row_data = rgb_pred.cpu().detach().numpy().tolist()[0]
                rgb_writer.writerow(row_data)







if __name__ == "__main__":
    num_classes = 300
    base_dir = os.path.abspath(os.path.dirname(__file__))
    datasets_list = ["CSL-300","CSL-500","CSL-1000"]
    weight_path = "D:/Desktop/server/TimeSformer-main/Swin/result/CSL-300/swin_epoch085.pth" # C:/uestc/code/TimeSformer-main/Swin/new_result/CSL-300/swin_epoch085.pth
    for dataset in datasets_list:
        dataset_path = os.path.join(base_dir,f"../datasets/{dataset}")
        if os.path.exists(dataset_path):
            for file in os.listdir(dataset_path):
                file_path = os.path.join(dataset_path,file)
                save_path = os.path.join(base_dir,f"features/RGB/{dataset}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_file_path = os.path.join(save_path,file)
                save_rgb_feature(save_file_path,file_path,num_classes,weight_path)


