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
        self.eta = 1

    def mixup(self, data, alpha=0.4, debug=False, weights=0.2, n_classes=6, device='cuda:0'):
        #data = data.to('cpu').detach().numpy().copy()
        #label = label.to('cpu').detach().numpy().copy()
        batch_size = len(data)
        weights = torch.from_numpy(np.random.uniform(low=0, high=alpha, size=batch_size)).to(device)
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index, :]
        x = [x1[i,:]*weights[i] + x2[i,:]*(1 - weights[i]) for i in range(batch_size)]
        x = torch.stack(x)
        return x

    def calc_mm_dist(self, x, centers):
        # x=(256, 640), centers=(6,640)
        dist = torch.mm(x, centers.T)
        return dist

    def forward(self, x, labels=None):
        batch_size = x.shape[0]
        #pseudo_outlier = self.mixup(x)
        # (256, 6, 640)
        self.x = F.normalize(x, dim=1)
        centers = F.normalize(self.centers ,dim=1)
        mm_dist_mat = self.calc_mm_dist(self.x, centers)
        dist_mat = (x.unsqueeze(1)-centers.unsqueeze(0)).pow(2)

        classes = torch.arange(self.num_class).long().cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        inlier_mask = labels.eq(classes.expand(batch_size, self.num_class))
        # (N, n_class, n_features)
        inlier_dist = dist_mat * inlier_mask.unsqueeze(2).float()
        mm_inlier_dist = mm_dist_mat * inlier_mask.float()
        # (N, 1, n_features)
        inlier_dist = inlier_dist.clamp(min=1e-12, max=1e+12).sum(dim=1)
        inlier_dist = inlier_dist.mean(dim=1)
        mm_inlier_dist = mm_inlier_dist.clamp(min=1e-12, max=1e+12).sum(dim=1)

        if self.training == True:
            outlier_mask = torch.logical_not(inlier_mask)
            # (N, n_class, n_features)
            outlier_dist = dist_mat * outlier_mask.unsqueeze(2).float()
            mm_outlier_dist = mm_dist_mat * outlier_mask.float()
            # (N, 1, n_features)
            outlier_dist = outlier_dist.clamp(min=1e-12, max=1e+12).sum(dim=1)
            outlier_dist = outlier_dist.mean(dim=1)
            mm_outlier_dist = mm_outlier_dist.clamp(min=1e-12, max=1e+12).sum(dim=1)
            # mixup
            #pseudo_outlier_dist = (pseudo_outlier.unsqueeze(1)-self.centers.unsqueeze(0)).pow(2)

            loss = (inlier_dist.mean() + -mm_inlier_dist.mean()) / (outlier_dist.mean() + -mm_outlier_dist.mean())
        else:
            loss = inlier_dist.mean()
        # anomaly_score = inlier_dist / (x.unsqueeze(1)-self.centers.unsqueeze(0)).pow(2).mean(dim=(1,2))
        
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
        #print(pred[:200])
        return loss, pred

    # def forward_classifier(self, x, section_label):
    #     x = self.effnet(x)
    #     print(x.shape)
    #     loss = self.effnet_loss(x, section_label)
    #     return loss, section_label
    
    # def forward_centerloss(self, x, section_label):
    #     loss, inlier_dist = self.centerloss_net(x, section_label)
    #     return loss, inlier_dist