import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torchaudio.transforms import Resample

import matplotlib.pyplot as plt

class CenterLoss(nn.Module):
    def __init__(self, num_class=10, num_feature=2, center_label=[0, 7, 14, 21, 28, 35]):
        super(CenterLoss, self).__init__()
        self.center_label = center_label
        self.num_class = num_class
        self.num_feature = num_feature
        self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))
        #self.k = 0.1

    def forward(self, x, labels=None):
        batch_size = x.shape[0]
        # (256, 6, 640)
        dist_mat = (x.unsqueeze(1)-self.centers.unsqueeze(0)).pow(2)
        
        classes = torch.arange(self.num_class).long().cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        inlier_mask = labels.eq(classes.expand(batch_size, self.num_class))
        # (N, n_class, n_features)
        inlier_dist = dist_mat * inlier_mask.unsqueeze(2).float()
        # (N, 1, n_features)
        inlier_dist = inlier_dist.clamp(min=1e-12, max=1e+12).mean(dim=1)
        inlier_dist = inlier_dist.mean(dim=1)

        if self.training == True:
            outlier_mask = torch.logical_not(inlier_mask)
            outlier_dist = dist_mat * outlier_mask.unsqueeze(2).float()
            outlier_dist = outlier_dist.clamp(min=1e-12, max=1e+12).mean(dim=1)
            outlier_dist = outlier_dist.mean(dim=1)
            loss = inlier_dist.mean() / outlier_dist.mean()
        else:
            loss = inlier_dist.mean()

        #inlier_dist = dist[]
        #outlier_dist = 
        return loss, inlier_dist

class FC_block(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(FC_block, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.silu = nn.SiLU()
        #self.ln1 = nn.LayerNorm(out_features)
    
    def forward(self, input, ):
        x = input
        x = self.silu(self.bn1(self.fc1(x)))
        return x

class CenterLossNet(nn.Module):
    def __init__(self, in_features, out_features=6, mid_features=None):
        
        super(CenterLossNet, self).__init__()
        if mid_features == None:
            mid_features = int(in_features / 2)        
        self.n_layers = 3
        self.fc_in = FC_block(in_features, mid_features)
        self.fc_blocks = nn.Sequential(
            *([FC_block(mid_features, mid_features)] * self.n_layers))
        self.cl_out = CenterLoss(num_class=out_features, num_feature=mid_features)

    def forward(self, x, section_label):
        x = self.fc_in(x)
        x = self.fc_blocks(x)
        loss, inlier_dist = self.cl_out(x, section_label)
        return loss, inlier_dist

from torch.nn import Parameter
import math
class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, self.s * torch.exp(logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            # print(B_avg)
            theta_med = torch.median(theta)
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        # print(self.s)
        output *= self.s

        return output

class EfficientNet_b1(nn.Module):
    def __init__(self, n_out=36, n_centers=6):
        super(EfficientNet_b1, self).__init__()
        #self.bn0 = nn.BatchNorm2d(128)
        #　モデルの定義
        self.effnet = timm.create_model('efficientnet_b1', pretrained=True)
        # forwardをover ride
        self.effnet.forward = self.effnet_forward
        self.fc_classifier = nn.Linear(1280, n_centers)
        self.fc0 = nn.Linear(1280, 1280, bias=False)
        self.bn0 = nn.BatchNorm1d(1280)
        self.swish = nn.SiLU()
        self.fc1 = nn.Linear(1280, 1280, bias=False)
        #　最終層の再定義
        #self.effnet.classifier = nn.Linear(1280, n_out)
        # 距離学習
        #self.adacos = AdaCos(1280, n_out)
        # section分類用のloss
        self.ClassifierLoss = nn.CrossEntropyLoss()
        self.CenterLoss = CenterLoss(n_centers, 1280)
        # CenterLoss用のネットワーク
        #self.centerloss_net = CenterLossNet(1280, n_centers)

    def effnet_forward(self, x):
        x = self.effnet.forward_features(x)
        x = self.effnet.global_pool(x)
        if self.effnet.drop_rate > 0.:
            x = F.dropout(x, p=self.effnet.drop_rate, training=self.training)
        return x

    def forward(self, x, section_label):
        # x = x.transpose(1, 2)
        # x = self.bn0(x)
        # x = x.transpose(1, 2)
        # if self.training == True:
        #     x, section_label = self.mixup(x, section_label)
        # else:
        #     batch_size, n_classes = x.shape[0], 6
        #     label_mat = torch.zeros((batch_size, n_classes, n_classes)).cuda()
        #     for i in range(batch_size):
        #         label_mat[i, section_label[i], section_label[i]] = 1  # onehot 
        #     # (classes: 0~35)
        #     section_label = torch.flatten(label_mat, start_dim=1, end_dim=-1).argmax(dim=1)
        
        # effnet
        embedding = self.effnet(x)
        #x = self.adacos(embedding, section_label)
        #classifier_loss = self.effnet_loss(x, section_label)
        classes_out = self.fc_classifier(embedding)
        centerloss_out = self.fc1(self.swish(self.bn0(self.fc0(embedding))))
        classifier_loss = self.ClassifierLoss(classes_out, section_label)
        center_loss, pred = self.CenterLoss(centerloss_out, section_label)
        loss = classifier_loss + center_loss
        
        return loss, pred

    # def forward_classifier(self, x, section_label):
    #     x = self.effnet(x)
    #     print(x.shape)
    #     loss = self.effnet_loss(x, section_label)
    #     return loss, section_label
    
    # def forward_centerloss(self, x, section_label):
    #     loss, inlier_dist = self.centerloss_net(x, section_label)
    #     return loss, inlier_dist