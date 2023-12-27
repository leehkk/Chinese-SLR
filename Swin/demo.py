import random

import cv2
import torch
import numpy as np
from torchvision.transforms import transforms


def frame_indices_tranform(video_length, sample_duration):
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
    return frame_indices

def center_crop(input_width, input_height, output_size):
    i = (input_width - output_size) // 2
    j = (input_height - output_size) // 2
    return i, j, i + output_size, j + output_size

path = "D:/Desktop/server/CSL-1000/0/0_1.mp4"
transform = transforms.Compose([transforms.Resize([112, 112]),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
video = cv2.VideoCapture(path)
video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
images = []
index_list = frame_indices_tranform(video_frames,16)
print(index_list)
# for i in range(self.frames):
for i in index_list:
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, image = video.read()
    if ret:
        height, width, _ = image.shape
        print(height,width)
        if height < width:
            image = cv2.resize(image, (320, 256))
        else:
            image = cv2.resize(image, (256, 320))
        new_height, new_width, _ = image.shape
        print(new_width,new_height)
        crop_box = center_crop(new_width, new_height, 224)

        # if flip_rand > 0.5:
        #     image = ImageOps.mirror(image)
        # image = transforms.functional.rotate(image, angle)
        image = image[crop_box[1]:crop_box[3],
                      crop_box[0]:crop_box[2]]
        print(image.shape)
        # image = image.crop(crop_box)
        image = torch.from_numpy(image).float()
        print(image.shape)
        assert image.shape[0] == 224 and image.shape[1] == 224
        image = image.permute(2,0,1)
        if transform is not None:
            image = transform(image)

    images.append(image)

images = torch.stack(images, dim=0)
# switch dimension for 3d cnn
images = images.permute(1, 0, 2, 3)
