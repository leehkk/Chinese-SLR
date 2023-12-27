import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from fvcore.common.file_io import PathManager
import cv2
from torchvision.models.video import Swin3D_T_Weights,Swin3D_S_Weights
"""
Implementation of Sign Language Dataset
"""


class WLASL(Dataset):
    def __init__(self,
                 path_label_dir,
                 frames=16,
                 num_classes=100,
                 split="train"):
        self.path_to_videos = []
        self.labels = []
        self.split = split
        self.frames = frames
        self.num_classes = num_classes
        self.sample_names = []
        self.path_to_file = os.path.join(path_label_dir, f"{self.split}.csv")
        with PathManager.open(self.path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (len(path_label.split(",")) == 2)
                path, label = path_label.split(",")
                self.path_to_videos.append(path)
                self.labels.append(int(label))
        assert (len(self.path_to_videos) >
                0), "Failed to load WLASL split {} from {}".format(
                    clip_idx, path_label_dir)

    def frame_indices_tranform(self, video_length, sample_duration):
        if video_length > sample_duration:
            random_start = random.randint(0, video_length - sample_duration)
            frame_indices = np.arange(random_start,
                                      random_start + sample_duration)
        else:
            frame_indices = np.arange(video_length)
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate(
                    (frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration]
            assert frame_indices.shape[0] == sample_duration
        return frame_indices

    def frame_indices_tranform_test(self,
                                    video_length,
                                    sample_duration,
                                    clip_no=0):
        if video_length > sample_duration:
            start = (video_length - sample_duration) // (self.test_clips -
                                                         1) * clip_no
            frame_indices = np.arange(start, start + sample_duration)
        elif video_length == sample_duration:
            frame_indices = np.arange(sample_duration)
        else:
            frame_indices = np.arange(video_length)
            while frame_indices.shape[0] < sample_duration:
                frame_indices = np.concatenate(
                    (frame_indices, np.arange(video_length)), axis=0)
            frame_indices = frame_indices[:sample_duration]

        return frame_indices

    def random_crop_paras(self, input_size, output_size):
        diff = input_size - output_size
        i = random.randint(0, diff)
        j = random.randint(0, diff)
        return i, j, i + output_size, j + output_size

    def center_crop(self, input_width, input_height, output_size):
        i = (input_width - output_size) // 2
        j = (input_height - output_size) // 2
        return i, j, i + output_size, j + output_size

    def read_images(self, folder_path, selected_frames):
        video = cv2.VideoCapture(folder_path)
        if not video.isOpened():
            print(folder_path," error")
        video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # step = video_frames // selected_frames
        # indices = np.arange(0, video_frames, step)[:selected_frames]
        indices = self.frame_indices_tranform(video_frames, selected_frames)
        frames = []
        for i in indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            if ret:
                frames.append(frame)
            else:
                while not ret:
                    video.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0,video_frames-1))
                    ret, frame = video.read()
                frames.append(frame)
        frames = np.array(frames)
        assert frames.shape[0] == 16
        # print(folder_path,indices.shape[0],frames.shape)
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        weights = Swin3D_S_Weights.DEFAULT
        preprocess = weights.transforms()
        frames = preprocess(frames_tensor)
        # print(images.shape)
        return frames

    def __len__(self):
        return len(self.path_to_videos)

    def __getitem__(self, idx):
        selected_folder = self.path_to_videos[idx]
        images = self.read_images(selected_folder, self.frames)
        label = torch.LongTensor([self.labels[idx]])
        if images.shape[1] < 16:
            print(selected_folder,images.shape)
        # print(images.size(), ', ', label.size())
        return {'data': images, 'label': label}
