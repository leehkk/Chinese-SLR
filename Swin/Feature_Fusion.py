import os
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from fvcore.common.file_io import PathManager
import cv2
import torch.nn as nn
"""
Implementation of Sign Language Dataset
"""


class Feature_Fusion(Dataset):
    def __init__(self,
                 path_to_feature,
                 path_to_label,
                 num_classes=300,
                 split="train"):
        self.key_list = []
        self.rgb_list = []
        self.labels = []
        self.split = split
        self.num_classes = num_classes
        self.path_to_key_file = os.path.join(path_to_feature, f"KEY/CSL-{num_classes}/{self.split}.csv")
        self.path_to_rgb_file = os.path.join(path_to_feature, f"RGB/CSL-{num_classes}/{self.split}.csv")
        self.path_to_label_file = os.path.join(path_to_label,f"CSL-{num_classes}/{self.split}.csv")
        with PathManager.open(self.path_to_label_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (len(path_label.split(",")) == 2)
                path, label = path_label.split(",")
                self.labels.append(int(label))
        with PathManager.open(self.path_to_key_file, "r") as f:
            for clip_idx, data in enumerate(f.read().splitlines()):
                if data == "":
                    continue
                data_list = data.split(",")
                data_list = [float(i) for i in data_list]
                data_tensor = torch.tensor(data_list)
                layer_norm = nn.LayerNorm(normalized_shape=(256,))
                data_tensor = layer_norm(data_tensor)
                data_list = data_tensor.detach().numpy()
                # print(data_list)
                self.key_list.append(data_list)
        with PathManager.open(self.path_to_rgb_file, "r") as f:
            for clip_idx, data in enumerate(f.read().splitlines()):
                if data == "":
                    continue
                data_list = data.split(",")
                data_list = [float(i) for i in data_list]
                self.rgb_list.append(data_list)
        rgb_tensor = torch.tensor(self.rgb_list)
        key_tensor = torch.tensor(self.key_list)
        self.input_tensor = torch.cat([rgb_tensor, key_tensor],dim=1)


    def __len__(self):
        return len(self.input_tensor)

    def __getitem__(self, idx):
        input_tensor = self.input_tensor[idx]
        label = torch.LongTensor([self.labels[idx]])
        # print(images.size(), ', ', label.size())
        return {'data': input_tensor, 'label': label}
