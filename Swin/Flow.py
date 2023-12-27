import csv
import os

import cv2
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import numpy as np
from fvcore.common.file_io import PathManager
from torchvision.models.video import Swin3D_T_Weights,Swin3D_S_Weights
"""
Implementation of Sign Language Dataset
"""


class Flow(Dataset):
    def __init__(self, data_path, frames=16, num_classes=500, split = "train", transform=None, test_clips=5,type = "Flow"):
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.frames = frames
        self.num_classes = num_classes
        self.test_clips = test_clips
        self.labels = []
        self.data_folder = []
        self.type = type
        self.path_to_file = os.path.join(data_path, f"{self.split}.csv")
        with open(self.path_to_file, "r") as f:
            content = csv.reader(f)
            for clip_idx, path_label in enumerate(content):
                path, label = path_label[0],path_label[1]
                self.data_folder.append(path)
                self.labels.append(int(label))
        print(f"load {len(self.data_folder)} videos")

    def frame_indices_tranform(self, video_length, sample_duration):
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

    def center_crop(self, input_width, input_height, output_size):
        i = (input_width - output_size) // 2
        j = (input_height - output_size) // 2
        return i, j, i + output_size, j + output_size

    def frame_indices_tranform_test(self, video_length, sample_duration, clip_no=0):
        if video_length > sample_duration:
            start = (video_length - sample_duration) // (self.test_clips - 1) * clip_no
            frame_indices = np.arange(start, start + sample_duration)
        elif video_length == sample_duration:
            frame_indices = np.arange(sample_duration) + 1
        else:
            frame_indices = np.arange(video_length)
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate((frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration]

        return frame_indices

    def random_crop_paras(self, input_size, output_size):
        diff = input_size - output_size
        i = random.randint(0, diff)
        j = random.randint(0, diff)
        return i, j, i + output_size, j + output_size

    def read_images(self, folder_path, clip_no=0,type="Flow"):
        files = os.listdir(folder_path)
        flow_frames = []
        rgb_frames = []
        for file in files:
            if file.split("_")[0] == "flow":
                flow_frames.append(os.path.join(folder_path,file))
            elif file.split("_")[0] == "rgb":
                rgb_frames.append(os.path.join(folder_path, file))
        if type == "Flow":
            frames_file = flow_frames
        elif type == "RGB":
            frames_file = rgb_frames
        # assert len(os.listdir(folder_path)) >= self.frames, "Too few images in your data folder: " + str(folder_path)
        video_length = len(frames_file)
        images = []

        index_list = self.frame_indices_tranform(video_length,self.frames)
        flip_rand = random.random()
        angle = (random.random() - 0.5) * 20
        crop_box = self.center_crop(270,480,224)
        for i in index_list:
            image = Image.open(frames_file[i])
            image = image.resize((270,480))
            if self.split == "train":
                if flip_rand > 0.5:
                    image = ImageOps.mirror(image)
                image = transforms.functional.rotate(image, angle)
                image = image.crop(crop_box)
                assert image.size[0] == 224
            else:
                crop_box = self.center_crop(270,480,224)
                image = image.crop(crop_box)
                # assert image.size[0] == 224
            images.append(image)
        frames = np.array(images)
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        weights = Swin3D_S_Weights.DEFAULT
        preprocess = weights.transforms()
        frames = preprocess(frames_tensor)
        # print(images.shape)
        # T, C, H, W
        # print(images.shape)
        return frames

    def __len__(self):
        return len(self.data_folder)

    def __getitem__(self, idx):
        selected_folder = self.data_folder[idx]
        #print(selected_folder)
        images = self.read_images(selected_folder,type=self.type)
        label = torch.LongTensor([self.labels[idx]])
        # print(images.size(), ', ', label.size())
        return {'data': images, 'label': label}
