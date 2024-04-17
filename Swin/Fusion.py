import math

import cv2
from fvcore.common.file_io import PathManager
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.video import swin3d_t, Swin3D_T_Weights,Swin3D_S_Weights,swin3d_s
import numpy as np
import torch
import os
from torchvision.transforms import transforms
import logging
from Swin.Feature_Fusion import Feature_Fusion
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from Swin.test_val import val_epoch
from Swin.train import train_epoch


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)

        self.attn_dropout = nn.Dropout(hidden_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(input_size, input_size)
        self.LayerNorm = LayerNorm(input_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        query_layer = self.query(input_tensor)
        key_layer = self.key(input_tensor)
        value_layer = self.value(input_tensor)


        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        print(hidden_states.shape)
        return hidden_states



#分类器
class FeatureConcatClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FeatureConcatClassifier, self).__init__()
        self.reshape = nn.Linear(768, 256, bias=False)
        self.query = nn.Linear(256, 256, bias=False)
        self.key = nn.Linear(256, 256, bias=False)
        self.value = nn.Linear(256, 256, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 4096, bias=False),
            nn.GELU(approximate='none'),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024, bias=False),
            nn.GELU(approximate='none'),
            nn.Dropout(0.5),
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256, bias=False),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes, bias=False)
        )

        self.classifier2 = nn.Linear(1024, num_classes, bias=False)
    def forward(self, x):
        self.token1 = x[: ,:768]
        self.token2 = x[:, 768:]
        self.token1 = self.reshape(self.token1)
        # energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # 分类
        mlp_output = self.mlp(x)
        output = self.classifier(mlp_output)
        return output


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

def Train_Classifier(model, criterion, optimizer, dataloader, device, epoch, logger, log_interval, writer):
    model.train()
    losses = []
    all_label = []
    all_pred = []
    for batch_idx, data in enumerate(dataloader):
        # get the inputs and labels
        inputs, labels = data['data'].to(device), data['label'].to(device)
        optimizer.zero_grad()
        # print(inputs.shape,labels.shape)
        # forward
        # print(inputs.shape)
        outputs = model(inputs.float())
        if isinstance(outputs, list):
            outputs = outputs[0]
        # compute the loss
        if labels.size(0) > 1:
            labels = labels.squeeze()
        else:
            labels = labels.squeeze()
            labels = labels.unsqueeze(0)
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        # compute the accuracy
        prediction = torch.max(outputs, 1)[1]
        all_label.extend(labels)
        all_pred.extend(prediction)
        if labels.size(0) > 1:
            score = accuracy_score(labels.cpu().data.squeeze().numpy(),
                                   prediction.cpu().data.squeeze().numpy())
        else:
            print(labels,prediction)
            if labels == prediction:
                score = 100.0
            else:
                score = 0.0
        # backward & optimize
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            logger.info("train_epoch {:3d} | iteration {:5d} | Loss {:.6f} | Acc {:.2f}%".format(epoch+1, batch_idx+1, loss.item(), score*100))


    # Compute the average loss & accuracy
    training_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    training_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'train': training_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'train': training_acc}, epoch+1)
    logger.info("Average Training Loss of Epoch {}: {:.6f} | Acc: {:.2f}%".format(epoch+1, training_loss, training_acc*100))


phase = 'Train'


# Use specific gpus
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = "CSL"
# Hyperparams
num_classes = 300 #100
epochs = 300
batch_size = 8
learning_rate = 0.001#1e-3 train, 1e-4 finetune
weight_decay = 1e-4 #1e-4 train
log_interval = 10
num_frames = 32
num_workers = 8
val_interval = 5
save_interval = 5
path_to_features = "D:/Desktop/server/TimeSformer-main/Swin/features"
path_to_csl = "D:/Desktop/server/TimeSformer-main/new_datasets"
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
        train_set = Feature_Fusion(path_to_feature=path_to_features,path_to_label=path_to_csl,
            num_classes=num_classes, split="train")
        val_set = Feature_Fusion(path_to_feature=path_to_features,path_to_label=path_to_csl,
            num_classes=num_classes, split="val")
        logger.info("CSL Train samples: {}".format(len(train_set)))
        logger.info("CSL Val samples: {}".format(len(val_set)))
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        test_set = Feature_Fusion(path_to_feature=path_to_features,path_to_label=path_to_csl,
                        num_classes=num_classes, split="test")
        val_set = Feature_Fusion(path_to_feature=path_to_features, path_to_label=path_to_csl,
                                 num_classes=num_classes, split="val")
        logger.info("CSL Test samples: {}".format(len(test_set)))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # 分类器
    model = FeatureConcatClassifier(num_classes)
    model.to(device)
    # print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = LabelSmoothingCrossEntropy()
    rgb_path = os.path.join(path_to_features,f"RGB/CSL-{num_classes}")
    key_path = os.path.join(path_to_features,f"KEY/CSL-{num_classes}")
    if phase == 'Train':
        criterion = LabelSmoothingCrossEntropy()
        logger.info("Training Started".center(60, '#'))
        for epoch in range(epochs):
            current_lr = scheduler.get_last_lr()[0]
            print('lr: ', current_lr)
            # Train the model
            Train_Classifier(model, criterion, optimizer, train_loader, device, epoch, logger, log_interval, writer)
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
        weight_path = os.path.join(result_path,"swin_epoch100.pth")
        test_loss = val_epoch(model, criterion, test_loader, device, 0, logger, writer, weight_path=weight_path,phase=phase)
        val_loss = val_epoch(model, criterion, val_loader, device, 0, logger, writer, weight_path=weight_path,
                              phase="Val")

    logger.info("Finished".center(60, '#'))