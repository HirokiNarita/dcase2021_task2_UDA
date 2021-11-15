import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

from metric_module import FocalLoss
from metric_module import ArcMarginProduct, AddMarginProduct

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import DropStripes
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

    def forward(self, x, pseudo_x=None, labels=None):
        batch_size = x.shape[0]
        # (256, 6, 640)
        dist_mat = (x.unsqueeze(1)-self.centers.unsqueeze(0)).pow(2)
        
        classes = torch.arange(self.num_class).long().cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        inlier_mask = labels.eq(classes.expand(batch_size, self.num_class))
        # (N, n_class, n_features)
        inlier_dist = dist_mat[:batch_size] * inlier_mask.unsqueeze(2).float()
        # (N, 1, n_features)
        inlier_dist = inlier_dist.clamp(min=1e-12, max=1e+12).sum(dim=1)
        inlier_dist = inlier_dist.mean(dim=1)

        if self.training == True:
            outlier_dist = (pseudo_x.unsqueeze(1)-self.centers.unsqueeze(0)).pow(2) # (N, 6, n_features)
            outlier_dist = outlier_dist.clamp(min=1e-12, max=1e+12).sum(dim=1)      # (N, 1, n_features)
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

class GDE(nn.Module):
    def __init__(self, n_classes=6, n_features=1280):
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
        #self.bn0 = nn.BatchNorm2d(128)
        self.n_classes = n_classes
        #　モデルの定義
        self.effnet = timm.create_model('efficientnet_b1', pretrained=True)
        # forwardをover ride
        self.effnet.forward = self.effnet_forward
        #self.fc_classifier = nn.Linear(1280, n_classes)
        self.n_views = 2
        self.device = 'cuda:0'
        self.temperature = 0.2        
        self.projection_head = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.SiLU(),
            )

        # self.fc_block0 = nn.Sequential(
        #     nn.Linear(1280, 1280, bias=False),
        #     nn.BatchNorm1d(1280),
        #     nn.SiLU(),
        #     nn.Linear(1280, 1280, bias=False),
        #     )

        self.arc_face = AddMarginProduct(in_features=1280,
                                         out_features=n_classes)
        
        # 128/4 = 32
        self.freq_dropper = DropStripes(dim=2, drop_width=16, 
            stripes_num=2)

        # # 64/8 = 8
        self.time_dropper = DropStripes(dim=3, drop_width=8, 
            stripes_num=2)

        self.forcal_loss = FocalLoss()
        
        self.criterion = nn.CrossEntropyLoss()

        self.gaussian_density_estimation = GDE(n_classes=6, n_features=1280)

        # section分類用のloss
        # self.ClassifierLoss = nn.CrossEntropyLoss()
        # self.CenterLoss = CenterLoss(n_centers, 1280)
        # CenterLoss用のネットワーク
        #self.centerloss_net = CenterLossNet(1280, n_centers)

    def effnet_forward(self, x):
        x = self.effnet.forward_features(x)
        x = self.effnet.global_pool(x)
        if self.effnet.drop_rate > 0.:
            x = F.dropout(x, p=self.effnet.drop_rate, training=self.training)
        return x

    def mixup(self, data, label, alpha=0.4, debug=False, weights=0.2, n_classes=6, device='cuda:0'):
        #data = data.to('cpu').detach().numpy().copy()
        #label = label.to('cpu').detach().numpy().copy()
        
        batch_size = len(data)
        weights = torch.from_numpy(np.random.uniform(low=0, high=alpha, size=batch_size)).to(device)
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index, :]
        x = [x1[i,:]*weights[i] + x2[i,:]*(1 - weights[i]) for i in range(batch_size)]
        
        x = torch.stack(x)
        return x

    def spec_cutmix(self, data, n_crop_freqs_high=64, device='cuda:0'):
        batch_size = data.shape[0]
        total_freq = data.shape[2]
        data = data.clone()
        # random pair
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index]
        # random freq range
        n_crop_freqs = torch.randint(low=1, high=n_crop_freqs_high, size=(batch_size,))
        bgn_freq = [torch.randint(low=0, high=total_freq - n_crop_freqs[i], size=(1,))[0] for i in range(batch_size)]
        bgn_freq = torch.stack(bgn_freq)
        # subst
        for i in range(batch_size):
            x1[i, :, bgn_freq[i]: bgn_freq[i]+n_crop_freqs[i], :] = x2[i, :, bgn_freq[i]: bgn_freq[i]+n_crop_freqs[i], :]
        
        return x1

    def info_nce_loss(self, features):
        
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def forward(self, x, section_label, label=None, device='cuda:0'):
        self.batch_size = x.shape[0]

        if self.training == True:
            x1 = self.time_dropper(self.freq_dropper(self.spec_cutmix(x))) # A1
            x2 = self.time_dropper(self.freq_dropper(self.spec_cutmix(x))) # A2
            x = torch.cat([x1, x2], dim=0)
            section_label = torch.cat([section_label, section_label], dim=0)

        # effnet
        embedding = self.effnet(x)

        x = self.projection_head(embedding)
        #pred_section = self.arc_face(embedding, section_label)
        #loss = self.forcal_loss(pred_section, section_label)
        
        pred_section = self.arc_face(embedding, section_label)
        loss = self.criterion(pred_section, section_label)

        if self.training == True:
            logits, labels = self.info_nce_loss(x)
            loss_clr = self.criterion(logits, labels)
            loss = loss + loss_clr
        

        pred = None
        if self.training == False:
            pred = self.gaussian_density_estimation.calc_distance(embedding, section_label)

        #print(pred[:200])
        return loss, embedding, pred

    # def forward_classifier(self, x, section_label):
    #     x = self.effnet(x)
    #     print(x.shape)
    #     loss = self.effnet_loss(x, section_label)
    #     return loss, section_label
    
    # def forward_centerloss(self, x, section_label):
    #     loss, inlier_dist = self.centerloss_net(x, section_label)
    #     return loss, inlier_dist