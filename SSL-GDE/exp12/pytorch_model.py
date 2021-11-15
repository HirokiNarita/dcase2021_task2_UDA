import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

from metric_module import FocalLoss
from metric_module import AdaCos

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation, DropStripes
from torchaudio.transforms import Resample

import matplotlib.pyplot as plt

class GDE(nn.Module):
    def __init__(self, n_classes=6, n_features=2096):
        super(GDE, self).__init__()
        self.n_classes = n_classes
        self.n_features = n_features

        self.mean = nn.Parameter(torch.zeros((n_classes, n_features)))
        self.inv_cov = nn.Parameter(torch.zeros(n_classes, n_features, n_features))

        self.mean.requires_grad = False
        self.inv_cov.requires_grad = False
    
    def set_param(self, X, section_label):
        for section in range(self.n_classes):
            idx = torch.where(section_label == section)[0]
            X_ = X[idx, :]
            self.mean[section, :] = X_.mean(dim=0)
            X_ = X_.detach().numpy().copy()
            I = np.identity(X_.shape[1])
            cov = np.cov(X_, rowvar=False) + 0.01 * I
            inv_cov = np.linalg.inv(cov)
            self.inv_cov[section, :, :] = torch.from_numpy(inv_cov)


    # https://github.com/bflammers/automahalanobis/blob/master/modules/mahalanobis.py
    def calc_distance(self, X, section_label):
        dists = []
        for section in range(self.n_classes):
            idx = torch.where(section_label == section)[0]
            X_ = X[idx, :]
            delta = X_ - self.mean[section, :]   # (N, feat) - (,feat) = (N, feat)
            
            dist = torch.mm(torch.mm(delta, self.inv_cov[section, :, :]), delta.t())  # (N, feat) * ((feat, feat) * (N, feat))
            dist = torch.diag(dist)
            dists.append(dist)
        dists = torch.cat(dists)
        return dists

class EfficientNet_b1(nn.Module):
    def __init__(self, n_out=36, n_classes=6):
        super(EfficientNet_b1, self).__init__()
        self.bn0 = nn.BatchNorm2d(128)
        self.n_classes = n_classes
        #　モデルの定義
        self.effnet = timm.create_model('efficientnet_b1', pretrained=True)
        # forwardをover ride
        self.effnet.forward = self.effnet_forward
        self.effnet.forward_features = self.forward_features

        # 128/4 = 32
        self.freq_dropper = DropStripes(dim=2, drop_width=16, 
            stripes_num=2)

        # 64/8 = 8
        self.time_dropper = DropStripes(dim=3, drop_width=8, 
            stripes_num=2)
        
        self.fc_block0 = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.BatchNorm1d(1280),
            nn.SiLU(),
            nn.Linear(1280, 1280),
            )

        self.arc_face = AdaCos(num_features=1280,
                               num_classes=n_classes)
        
        self.forcal_loss = FocalLoss()
        self.distrib_loss = nn.MSELoss()

        self.gaussian_density_estimation = GDE(n_classes=6, n_features=2096)

        # section分類用のloss
        # self.ClassifierLoss = nn.CrossEntropyLoss()
        # self.CenterLoss = CenterLoss(n_centers, 1280)
        # CenterLoss用のネットワーク
        #self.centerloss_net = CenterLossNet(1280, n_centers)

    def forward_features(self, x):
        features = []
        x = self.effnet.conv_stem(x)
        x = self.effnet.bn1(x)
        x = self.effnet.act1(x)
        features.append(F.adaptive_avg_pool2d(x, 1))
        for i, block_layer in enumerate(self.effnet.blocks):
            x = block_layer(x)
            features.append(F.adaptive_avg_pool2d(x, 1))
        #x = self.blocks(x)
        x = self.effnet.conv_head(x)
        x = self.effnet.bn2(x)
        x = self.effnet.act2(x)
        features.append(F.adaptive_avg_pool2d(x, 1))
        features = torch.cat(features, dim=1)

        return x, features

    def effnet_forward(self, x):
        x, feature = self.effnet.forward_features(x)
        x = self.effnet.global_pool(x)
        if self.effnet.drop_rate > 0.:
            x = F.dropout(x, p=self.effnet.drop_rate, training=self.training)
        return x, feature

    def mixup(self, data, label, alpha=0.4, debug=False, weights=0.2, n_classes=6, device='cuda:0'):
        #data = data.to('cpu').detach().numpy().copy()
        #label = label.to('cpu').detach().numpy().copy()
        X_src, X_tgt = torch.chunk(data, 2)
        batch_size = len(X_src)
        weights = torch.from_numpy(np.random.uniform(low=0, high=alpha, size=batch_size)).to(device) # これでよい？
        index = np.random.permutation(batch_size)
        x1, x2 = X_src, X_tgt
        y1, y2 = label, label[index]

        # src
        x = [x1[i,:]*weights[i] + x2[i,:]*(1 - weights[i]) for i in range(batch_size)]
        x = torch.stack(x)

        # labelが半分なので追加
        #y = torch.cat([y, y], dim=0)
        # src+tgt
        return x

    def forward(self, x, section_label, label=None, device='cuda:0', is_aug=True, eval=False):
        batch_size = x.shape[0]
        #print(section_label)
        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        # Data Aug
        if is_aug == True:
            x, y = self.mixup(x, section_label)
            #x = self.time_dropper(self.freq_dropper(x))
        # effnet
        embedding, features = self.effnet(x)
        features = features.squeeze()
        if is_aug == True:
            # 分類loss
            # L_adapt
            #embedding = self.fc_block0(embedding)
            Z_src, Z_tgt = torch.chunk(embedding, 2)
            L_adapt = self.distrib_loss(Z_src, Z_tgt)
            Z_src = self.fc_block0(Z_src)
            pred_section = self.arc_face(Z_src, y)
            loss = self.forcal_loss(pred_section, y)
            #L_adapt = self.distrib_loss(Z_src, Z_tgt)
            loss = loss + L_adapt
        else:
            y = F.one_hot(section_label, num_classes=self.n_classes)
            embedding = self.fc_block0(embedding)
            pred_section = self.arc_face(embedding, y)
            loss = self.forcal_loss(pred_section, y)

        pred = None
        if eval == True:
            pred = self.gaussian_density_estimation.calc_distance(features, section_label)

        #print(pred[:200])
        return loss, features, pred

    # def forward_classifier(self, x, section_label):
    #     x = self.effnet(x)
    #     print(x.shape)
    #     loss = self.effnet_loss(x, section_label)
    #     return loss, section_label
    
    # def forward_centerloss(self, x, section_label):
    #     loss, inlier_dist = self.centerloss_net(x, section_label)
    #     return loss, inlier_dist