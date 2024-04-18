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
from CSL_S import CSL
import torch.nn.functional as F

from test_val import val_epoch
# from Swin.train import train_epoch
from train_nospl import train_epoch


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

    def forward(self, input, target):
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
    def __init__(self, n_samples=0,threshold = 0.5,growing_factor = 0.1):
        super(SPLLoss, self).__init__()
        self.samples = n_samples
        self.threshold = threshold
        self.growing_factor = growing_factor
        self.v = torch.zeros(self.samples, requires_grad=False)
        self.train_samples = int(self.samples * self.threshold)
        self.v[:self.train_samples] = 1.0

    def forward(self, input, target, index, device):
        criterion = nn.CrossEntropyLoss(reduction='none')
        super_loss = criterion(input, target)
        temp_v = torch.zeros(len(index), requires_grad=False)
        # temp_v = (super_loss < self.threshold).int()
        for i in index:
            pos = i % len(index)
            temp_v[pos] = self.v[i].item()
        return torch.mean(super_loss * temp_v.float().to(device))

    def increase_threshold(self):
        self.threshold = self.threshold + self.growing_factor
        self.train_samples = int(self.samples * self.threshold)
        if self.train_samples >= self.samples:
            self.v[:] = 1.0
        else:
            self.v[:self.train_samples] = 1.0
        # self.threshold *= (1 + self.growing_factor)

    def get_samples(self):
        return torch.sum(self.v)

    def init_v(self):
        self.v = torch.zeros(self.samples, requires_grad=False)

phase = 'Train'


# Use specific gpus
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = "CSL"
# Hyperparams
num_classes = 300 #100
epochs = 100
batch_size = 8
learning_rate = 0.005#1e-3 train, 1e-4 finetune
weight_decay = 1e-4 #1e-4 train
log_interval = 10
num_frames = 32
num_workers = 8
val_interval = 5
save_interval = 5
path_to_csl = "C:/uestc/code/TimeSformer-main/new_datasets/{}-{}".format(dataset, num_classes)
result_path = "C:/uestc/code/TimeSformer-main/Swin/new_result/{}-{}".format(dataset, num_classes)
log_path = "C:/uestc/code/TimeSformer-main/Swin/new_log/{}-{}/train.log".format(dataset, num_classes)
sum_path = "C:/uestc/code/TimeSformer-main/Swin/run/{}-{}".format(dataset, num_classes)
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
    model.head = nn.Linear(768, num_classes)
    # print(model)
    model = model.to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    if phase == 'Train':
        # criterion = SPLLoss(len(train_set))
        criterion = LabelSmoothingCrossEntropy()
        logger.info("Training Started".center(60, '#'))
        for epoch in range(epochs):
            current_lr = scheduler.get_last_lr()[0]
            print('lr: ', current_lr)
            # Train the model
            # criterion.init_v()
            train_epoch(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)
            # spl:
            # if epoch >= 10:
            #     if epoch % 5 == 0:
            #         criterion.increase_threshold()
            # train_samples = criterion.get_samples()
            #logger.info("epoch {} train samples: {}".format(epoch, train_samples))
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
        best_weight_path = ""
        best_acc = 0.0
        best_top_5 = 0.0
        for weight in os.listdir(result_path):
            weight_path = os.path.join(result_path, weight)
            logger.info(f"Testing Started".center(60, '#'),fr"weight path:{weight_path}")
            test_loss ,acc , top5_acc = val_epoch(model, criterion, test_loader, device, 0, logger, writer, weight_path=weight_path,phase=phase)
            if acc > best_acc:
                best_weight_path = weight_path
                best_acc = acc
                best_top_5 = top5_acc
        # val_loss = val_epoch(model, criterion, val_loader, device, 0, logger, writer, weight_path=weight_path,phase="Val")
        logger.info(f"best weight_path:{best_weight_path},best_acc:{best_acc},best_top5:{top5_acc}".center(60, '#'))
    logger.info("Finished".center(60, '#'))
