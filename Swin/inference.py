import csv
import random

import cv2
from torch import nn

from torchvision.models.video import Swin3D_S_Weights,swin3d_s
import numpy as np
import torch
import os

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









if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 300
    base_dir = os.path.abspath(os.path.dirname(__file__))
    weight_path = os.path.join(base_dir,"result/best_swin_epoch085.pth")
    rgb_model = swin3d_s(weights=None)
    rgb_model.head = nn.Linear(768, num_classes)
    state_dict = torch.load(weight_path)
    rgb_model.load_state_dict(state_dict)
    rgb_model = rgb_model.to(device)
    label_path = os.path.join(base_dir,"tools/class.csv")
    dataset_path = os.path.join(base_dir,"demo")
    data_dict = {}
    with PathManager.open(label_path, "r") as f:
        for index, data in enumerate(f.read().splitlines()):
            class_name, class_index = data.split(",")
            if class_name != "" or class_index != "":
                data_dict[class_index] = class_name
    for file in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file)
        rgb_frames = swin_get_rgb(file_path).unsqueeze(0).to(device)
        rgb_output = rgb_model(rgb_frames)
        rgb_pred = torch.max(rgb_output, 1)[1].item()
        print(data_dict[str(rgb_pred)])



