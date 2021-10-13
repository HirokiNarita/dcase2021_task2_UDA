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
        self.k = 0.5

    def replace_label(self, labels, outlier_num=99):
        # 7の倍数を置き換え（もっと適切な方法があるはず）
        # 他の値を99で置き換え
        #labels = labels.to('cpu').detach().clone()
        labels = torch.where((labels % 7) == 0 , labels, outlier_num)
        for i in range(len(self.center_label)):
            labels[labels == self.center_label[i]] = i

        return labels

    def forward(self, x, labels=None):
        if labels == None:
            labels = torch.zeros(x.shape[0]).long().cuda()
        # labels <= 6
        # replace label [0,7,14,21,28,35]:[0,1,2,3,4,5]
        # , outlier=(6)
        labels = self.replace_label(labels)
        # inlier用のcenter行列作成
        inlier_idx = (labels < 7).nonzero()
        inlier_x, inlier_label = x[inlier_idx], labels[inlier_idx]
        inlier_center = self.centers[inlier_label]
        inlier_dist = (inlier_x-inlier_center).pow(2)
        # outlier用のcenter行列作成
        # 全centerから遠く
        if self.training == True:
            outlier_idx = (labels == 99).nonzero()
            outlier_x, outlier_label = x[outlier_idx], labels[outlier_idx]
            outlier_x = x[outlier_idx]
            # 怪しい
            outlier_dist = (outlier_x-self.centers.unsqueeze(0)).pow(2)
            loss = inlier_dist.mean() - self.k * outlier_dist.mean()
        else:
            loss = inlier_dist.mean()

        inlier_dist = inlier_dist.mean(-1)
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

class EfficientNet_b1(nn.Module):
    def __init__(self, n_out=36, n_centers=6):
        super(EfficientNet_b1, self).__init__()
        self.bn0 = nn.BatchNorm2d(128)
        #モデルの定義
        self.effnet = timm.create_model('efficientnet_b1', pretrained=True)
        # forwardをover ride
        self.effnet.forward = self.effnet_forward
        #最終層の再定義
        self.effnet.classifier = nn.Linear(1280, n_out)
        # section分類用のloss
        self.effnet_loss = nn.CrossEntropyLoss()

        # CenterLoss用のネットワーク
        self.centerloss_net = CenterLossNet(1792, n_centers)
    
    def mixup(self, data, label, alpha=1, debug=False, weights=0.6, n_classes=6, device='cuda:0'):
        #data = data.to('cpu').detach().numpy().copy()
        #label = label.to('cpu').detach().numpy().copy()
        batch_size = len(data)
        label_mat = torch.zeros((batch_size, n_classes, n_classes)).to(device)    # (N, C_n, C_n)
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index]
        y1, y2 = label, label[index]
        x = torch.cat([
            torch.unsqueeze(
                x1[i,:,:,:]*weights + x2[i,:,:,:]*(1 - weights),
                0) \
                for i in range(batch_size)],
                dim=0)
        # onehot 2d matrix (batch, 6, 6) => onehot vector (batch, 36) => index vector (batch, 1)
        for i in range(batch_size):
            label_mat[i, y1[i], y2[i]] = 1  # onehot
        # (classes: 0~35)    
        label = torch.flatten(label_mat, start_dim=1, end_dim=-1).argmax(dim=1)
        
        return x, label

    def effnet_forward(self, x):
        x = self.effnet.forward_features(x)
        x = self.effnet.global_pool(x)
        if self.effnet.drop_rate > 0.:
            x = F.dropout(x, p=self.effnet.drop_rate, training=self.training)
        return x
       
    def forward_classifier(self, x, section_label):
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)
        if self.training == True:
            x, section_label = self.mixup(x, section_label)
        else:
            batch_size, n_classes = x.shape[0], 6
            label_mat = torch.zeros((batch_size, n_classes, n_classes)).cuda()
            for i in range(batch_size):
                label_mat[i, section_label[i], section_label[i]] = 1  # onehot 
            # (classes: 0~35)
            section_label = torch.flatten(label_mat, start_dim=1, end_dim=-1).argmax(dim=1)
        x = self.effnet(x)
        loss = self.effnet_loss(x, section_label)
        return loss, section_label
    
    def forward_centerloss(self, x, section_label):
        loss, inlier_dist = self.centerloss_net(x, section_label)
        return loss, inlier_dist