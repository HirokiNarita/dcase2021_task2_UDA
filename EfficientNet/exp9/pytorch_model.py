import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from scipy.spatial.distance import cdist, cosine

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
        self.t = 0.07

    def replace_label(self, labels, outlier_num=99):
        # 7の倍数を置き換え（もっと適切な方法があるはず）
        # 他の値を99で置き換え
        #labels = labels.to('cpu').detach().clone()
        labels = torch.where((labels % 7) == 0 , labels, outlier_num)
        for i in range(len(self.center_label)):
            labels[labels == self.center_label[i]] = i

        return labels

    def cosine_similarity(self, X1, X2):
        X1 = F.normalize(X1, dim=1)
        X2 = F.normalize(X2, dim=1)
        sim_mtx = torch.mm(X1, X2.T)
        return sim_mtx
        

    def forward(self, x, labels=None):
        #x = F.normalize(x)
        # labels <= 6
        # replace label [0,7,14,21,28,35]:[0,1,2,3,4,5]
        # , outlier=(6)
        labels = self.replace_label(labels).to('cpu').detach().clone()
        #print('labels', labels)
        # inlier用のcenter行列作成
        inlier_idx = (labels < 7).nonzero().to('cpu')
        inlier_x, inlier_label = x[inlier_idx].squeeze(), labels[inlier_idx]
        inlier_center = self.centers[inlier_label].squeeze()
        #print(f'inlier_x:{inlier_x.shape}, inlier_center:{inlier_center.shape}')

        inlier_dist = self.cosine_similarity(
            inlier_x.squeeze(),
            inlier_center.squeeze(),
            )
        #print(f'inlier_dist:{inlier_dist.shape}')
        #print(inlier_dist)
        #inlier_dist = (inlier_x-inlier_center).pow(2)
        # outlier用のcenter行列作成
        # 全centerから遠く
        if self.training == True:
            outlier_idx = (labels == 99).nonzero().to('cpu')
            outlier_x, outlier_label = x[outlier_idx].squeeze(), labels[outlier_idx]
            # 怪しい
            #print(f'outlier_x:{outlier_x.shape}, outlier_center:{self.centers.shape}')
            outlier_dist = self.cosine_similarity(outlier_x, self.centers)
            #print(f'outlier_dist:{outlier_dist.shape}')
            #print(outlier_dist)
            loss = - torch.log(torch.exp(inlier_dist.mean()/self.t) / torch.exp(outlier_dist.mean()/self.t))
        else:
            loss = - torch.log(torch.exp(inlier_dist.mean()/self.t))

        return loss, inlier_dist.mean(-1)

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
        self.n_layers = 2
        self.fc_in = FC_block(in_features, mid_features)
        self.fc_blocks = FC_block(mid_features, mid_features)
        self.fc0 = nn.Linear(mid_features, mid_features)
        
        self.cl_out = CenterLoss(num_class=out_features, num_feature=mid_features)

    def forward(self, x, section_label):
        #x = F.normalize(x)
        x = self.fc_in(x)
        x = self.fc_blocks(x)
        x = self.fc0(x)
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
        self.bn0 = nn.BatchNorm2d(128)
        #　モデルの定義
        self.effnet = timm.create_model('efficientnet_b1', pretrained=True)
        # forwardをover ride
        self.effnet.forward = self.effnet_forward
        #　最終層の再定義
        #self.effnet.classifier = nn.Linear(1280, n_out)
        # 距離学習
        self.adacos = AdaCos(1280, n_out)
        # section分類用のloss
        self.effnet_loss = nn.CrossEntropyLoss()

        # CenterLoss用のネットワーク
        self.centerloss_net = CenterLossNet(1280, n_centers)

    def effnet_forward(self, x):
        x = self.effnet.forward_features(x)
        x = self.effnet.global_pool(x)
        if self.effnet.drop_rate > 0.:
            x = F.dropout(x, p=self.effnet.drop_rate, training=self.training)
        return x
    
    def mixup(self, data, label, alpha=1, debug=False, weights=0.4, n_classes=6, device='cuda:0'):
        #data = data.to('cpu').detach().numpy().copy()
        #label = label.to('cpu').detach().numpy().copy()
        batch_size = len(data)
        label_mat = torch.zeros((batch_size, n_classes, n_classes))    # (N, C_n, C_n)
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
        label_mat = torch.flatten(label_mat, start_dim=1, end_dim=-1).argmax(dim=1)
        
        return x, label_mat.cuda()

    def forward(self, x, section_label):
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
        
        # effnet
        embedding = self.effnet(x)
        x = self.adacos(embedding, section_label)
        classifier_loss = self.effnet_loss(x, section_label)
        center_loss, pred = self.centerloss_net(embedding, section_label)
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