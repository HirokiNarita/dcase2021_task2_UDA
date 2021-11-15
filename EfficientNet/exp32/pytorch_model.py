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
        inlier_dist = inlier_dist.clamp(min=1e-12, max=1e+12).sum(dim=1)
        inlier_dist = inlier_dist.mean(dim=1)

        if self.training == True:
            outlier_mask = torch.logical_not(inlier_mask)
            # (N, n_class, n_features)
            outlier_dist = dist_mat * outlier_mask.unsqueeze(2).float()
            # (N, 1, n_features)
            outlier_dist = outlier_dist.clamp(min=1e-12, max=1e+12).sum(dim=1)
            outlier_dist = outlier_dist.mean(dim=1)
            loss = inlier_dist.mean() / self.eta*outlier_dist.mean()
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
        self.n_centers = out_features
        self.n_layers = 1
        
        self.fc0 = nn.Linear(1280, 1280, bias=False)
        self.bn0 = nn.BatchNorm1d(1280)
        self.act = nn.SiLU()

        self.fc_out = [nn.Linear(1280, 1280, bias=False)]*out_features
        self.fc_out = nn.ModuleList(self.fc_out)

        self.cl_out = CenterLoss(out_features, 1280)

    def forward(self, x, section_label):
        x = self.fc0(x)
        x = self.bn0(x)
        x = self.act(x)
        centerloss_out = []
        for i in range(self.n_centers):
            idx = torch.where(section_label == i)
            centerloss_out.append(self.fc_out[i](x[idx]))
        centerloss_out = torch.cat(centerloss_out, dim=0)
        
        loss, inlier_dist = self.cl_out(centerloss_out, section_label)
        return loss, inlier_dist

class VAE(nn.Module):
    def __init__(self, in_features, mid_features=512):
        
        super(VAE, self).__init__()
        self.beta = 1
        self.kld_weight = 256 / 6000

        self.in_features = in_features
        self.mid_features = mid_features
        
        self.fc0 = nn.Linear(in_features, in_features)
        
        # enc
        self.fc1 = nn.Linear(in_features, mid_features)
        #self.bn1 = nn.BatchNorm1d(mid_features)

        # latent
        self.fc_mean = nn.Linear(mid_features, mid_features)
        self.fc_var = nn.Linear(mid_features, mid_features)
        
        # dec
        self.fc2 = nn.Linear(mid_features, in_features)
        #self.bn2 = nn.BatchNorm1d(in_features)
        
        # out
        self.fc3 = nn.Linear(in_features, in_features)
        
        self.act = nn.SiLU()

    def encoder(self, x):
        x = self.act(self.fc1(x))
        mean = self.fc_mean(x) # 平均
        log_var = self.fc_var(x) # 分散の対数
        return mean, log_var

    # 潜在ベクトルのサンプリング(再パラメータ化)
    def reparametrizaion(self, mean, log_var, device):
        epsilon = torch.randn(mean.shape, device=device)
        return mean + epsilon*torch.exp(0.5 * log_var)

    # デコーダー
    def decoder(self, z):
        y = self.act(self.fc2(z))
        y = self.fc3(y)
        return y

    def forward(self, x, device):
        x = self.fc0(x)
        mean, log_var = self.encoder(x)
        z = self.reparametrizaion(mean, log_var, device)
        x_hat = self.decoder(z)
        
        recons_loss =F.mse_loss(x_hat, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + self.beta * self.kld_weight * kld_loss
        
        return loss , z, x_hat

class EfficientNet_b1(nn.Module):
    def __init__(self, n_out=36, n_centers=6):
        super(EfficientNet_b1, self).__init__()
        #self.bn0 = nn.BatchNorm2d(128)
        #　モデルの定義
        self.n_centers = n_centers
        self.effnet = timm.create_model('efficientnet_b1', pretrained=True)
        # forwardをover ride
        self.effnet.forward = self.effnet_forward
        self.vae = VAE(in_features=1280)
        #self.fc_classifier = nn.Linear(1280, n_centers)
        self.cl_net = CenterLossNet(1280, out_features=6)
        #　最終層の再定義
        #self.effnet.classifier = nn.Linear(1280, n_out)
        # 距離学習
        #self.adacos = AdaCos(1280, n_out)
        # section分類用のloss
        #self.ClassifierLoss = nn.CrossEntropyLoss()
        #self.CenterLoss = CenterLoss(n_centers, 1280)
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
        lower_bound , _, x_hat = self.vae(embedding, device='cuda:0')
   
        #classifier_loss = self.ClassifierLoss(classes_out, section_label)
        center_loss, pred = self.cl_net(x_hat, section_label)
        loss = center_loss + lower_bound
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