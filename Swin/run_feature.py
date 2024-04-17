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

from Swin.test_val import val_epoch
from Swin.train import train_epoch


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

class SPLCrossEntropyLoss(nn.Module):
    def __init__(self, threshold=0.000000001, growing_factor=1.35):
        super(SPLCrossEntropyLoss, self).__init__()
        self.threshold = threshold  # 初始化阈值
        self.growing_factor = growing_factor  # 初始化增长因子

    def forward(self, input, target, v=None):
        # 计算交叉熵损失
        ce_loss = nn.functional.cross_entropy(input, target)

        # if v is None:
        #     # 如果未提供v向量，则默认为全1向量，表示所有样本的权重都为1
        #     v = torch.ones_like(ce_loss)

        # 计算SPL值
        v = ce_loss < self.threshold

        # 根据SPL值更新阈值
        self.threshold *= self.growing_factor

        # 计算加权后的损失
        weighted_loss = ce_loss * v

        # 返回加权后的损失
        return torch.mean(weighted_loss)


class SPLLoss(nn.Module):
    def __init__(self, n_samples=0,threshold = 0.000000001,growing_factor = 2):
        super(SPLLoss, self).__init__()
        self.threshold = threshold
        self.growing_factor = growing_factor
        self.v = torch.zeros(n_samples, requires_grad=False)

    def forward(self, input, target, index):
        super_loss = nn.functional.cross_entropy(input, target)
        temp_v = super_loss < self.threshold
        j = 0
        for i in index:
            self.v[i] = temp_v[j].numpy()
            j += 1
        return torch.mean(super_loss * temp_v.float())

    def increase_threshold(self):
        self.threshold *= self.growing_factor



phase = 'Train'


# Use specific gpus
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = "CSL"
# Hyperparams
num_classes = 100 #100
epochs = 100
batch_size = 8
learning_rate = 0.005#1e-3 train, 1e-4 finetune
weight_decay = 1e-4 #1e-4 train
log_interval = 10
num_frames = 16
num_workers = 1
val_interval = 5
save_interval = 5
path_to_csl = "D:/Desktop/server/TimeSformer-main/datasets/{}-{}".format(dataset, num_classes)
result_path = "D:/Desktop/server/TimeSformer-main/Swin/result/{}-{}".format(dataset, num_classes)
log_path = "D:/Desktop/server/TimeSformer-main/Swin/log/{}-{}/train.log".format(dataset, num_classes)
sum_path = "D:/Desktop/server/TimeSformer-main/Swin/run/{}-{}".format(dataset, num_classes)
# Log to file & tensorboard writer
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
logger = logging.getLogger('SLR')
logger.info('Logging to file...')
writer = SummaryWriter(sum_path)
# Train with 3DCNN
if __name__ == '__main__':
    if phase == "Train":
        train_set = CSL(path_label_dir=path_to_csl, frames=num_frames,
            num_classes=num_classes, split="train")
        val_set = CSL(path_label_dir=path_to_csl, frames=num_frames,
            num_classes=num_classes, split="val")
        logger.info("CSL Train samples: {}".format(len(train_set)))
        logger.info("CSL Val samples: {}".format(len(val_set)))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_set = CSL(path_label_dir=path_to_csl, frames=num_frames,
                        num_classes=num_classes, split="test")
        logger.info("CSL Test samples: {}".format(len(test_set)))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=True)
        val_set = CSL(path_label_dir=path_to_csl, frames=num_frames,
            num_classes=num_classes, split="val")
        logger.info("CSL Val samples: {}".format(len(val_set)))
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model = swin3d_s(weights='KINETICS400_V1')

    model.head = nn.Sequential()
    model.norm = nn.Sequential()
    model.avgpool = nn.Sequential()
    print(model)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    if phase == 'Train':
        criterion = SPLLoss(len(train_set))
        logger.info("Training Started".center(60, '#'))
        for epoch in range(epochs):
            current_lr = scheduler.get_last_lr()[0]
            print('lr: ', current_lr)
            # Train the model
            train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)
            scheduler.step()
            # Validate the model
            if (epoch + 1) % val_interval == 0:

                val_loss = val_epoch(model, criterion, val_loader, device, epoch, logger, writer)


            # Save model
            if (epoch + 1) % save_interval == 0:
                torch.save(model.state_dict(),
                           os.path.join(result_path, "swin_epoch{:03d}.pth".format(epoch + 1)))
                logger.info("Epoch {} Model Saved".format(epoch + 1).center(60, '#'))
    elif phase == 'Test':
        criterion = LabelSmoothingCrossEntropy()
        logger.info("Testing Started".center(60, '#'))
        weight_path = os.path.join(result_path,"swin_epoch020.pth")
        test_loss = val_epoch(model, criterion, test_loader, device, 0, logger, writer, weight_path=weight_path,phase=phase)
        val_loss = val_epoch(model, criterion, val_loader, device, 0, logger, writer, weight_path=weight_path,
                              phase="Val")

    logger.info("Finished".center(60, '#'))
