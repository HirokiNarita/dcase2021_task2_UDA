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
        # (256, 6, feat)
        dist_mat = (x.unsqueeze(1)-self.centers.unsqueeze(0)).pow(2)
        # (256, 6, 1)
        labels = labels.unsqueeze(-1)
        # (N, n_class, n_features)
        inlier_dist = dist_mat * labels
        # (N, 1, n_features)
        inlier_dist = inlier_dist.clamp(min=1e-12, max=1e+12).sum(dim=1)
        # (N, 1, 1)
        inlier_dist = inlier_dist.mean(dim=1)

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

class Autoencoder(nn.Module):
    def __init__(self, in_features=1280, mid_features=320, n_classes=6):
        
        super(Autoencoder, self).__init__()
        
        self.enc = nn.Sequential(
            FC_block(in_features+n_classes, 640),
            FC_block(640, 640),
            FC_block(640, mid_features),
            )
        self.dec = nn.Sequential(
            FC_block(mid_features+n_classes, 640),
            FC_block(640, 640),
            FC_block(640, in_features),
            )
        #self.ln1 = nn.LayerNorm(out_features)
    
    def forward(self, input, label):
        x = torch.cat([input, label.float()], dim=1)
        x = self.enc(x)
        x = torch.cat([x, label.float()], dim=1)
        x = self.dec(x)

        return x

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

        self.ae = Autoencoder()
        # # 128/4 = 32
        # self.freq_dropper = DropStripes(dim=2, drop_width=16, 
        #     stripes_num=2)

        # # # 64/8 = 8
        # self.time_dropper = DropStripes(dim=3, drop_width=8, 
        #     stripes_num=2)
        
        # self.fc_block = nn.Sequential(
        #     nn.Linear(2096, 2096),
        #     nn.BatchNorm1d(2096),
        #     nn.SiLU(),
        #     nn.Linear(2096, 2096),
        #     )

        # self.arc_face = AdaCos(num_features=1280,
        #                        num_classes=n_classes)
        
        #self.forcal_loss = FocalLoss()

        #self.gaussian_density_estimation = GDE(n_classes=6, n_features=2096)

        # section分類用のloss
        # self.ClassifierLoss = nn.CrossEntropyLoss()
        self.CenterLoss = CenterLoss(self.n_classes, 2096)
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
        
        batch_size = len(data)
        weights = torch.from_numpy(np.random.uniform(low=0, high=alpha, size=batch_size)).to(device)
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index, :]
        y1, y2 = label, label[index]
        y1, y2 = F.one_hot(y1, num_classes=self.n_classes), F.one_hot(y2, num_classes=self.n_classes)
        x = [x1[i,:]*weights[i] + x2[i,:]*(1 - weights[i]) for i in range(batch_size)]
        x = torch.stack(x)
        y = [y1[i,:]*weights[i] + y2[i,:]*(1 - weights[i]) for i in range(batch_size)]
        y = torch.stack(y)
        return x, y

    def forward(self, x, section_label, label=None, device='cuda:0', is_aug=True, eval=False):
        batch_size = x.shape[0]

        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        # Data Aug
        if is_aug == True:
            x, y = self.mixup(x, section_label)
        else:
            y = F.one_hot(section_label, num_classes=self.n_classes)
        # effnet
        embedding, features = self.effnet(x)
        features = features.squeeze()

        ae_out = self.ae(embedding, y)
        #classifier_loss = self.ClassifierLoss(classes_out, section_label)

        #center_loss, pred = self.CenterLoss(centerloss_out, y)
        loss = F.mse_loss(embedding, ae_out) #center_loss
        pred = (embedding - ae_out).pow(2).mean(dim=1)
        # pred = None
        # if eval == True:
        #     pred = self.gaussian_density_estimation.calc_distance(features, section_label)

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