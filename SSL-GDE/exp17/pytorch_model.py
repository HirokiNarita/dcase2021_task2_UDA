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
        self.bn0 = nn.BatchNorm2d(224)
        #self.bn1 = nn.BatchNorm2d(128)
        self.n_classes = n_classes
        #　モデルの定義
        self.effnet = timm.create_model('efficientnet_b1', pretrained=True)
        # forwardをover ride
        self.effnet.forward = self.effnet_forward
        self.effnet.forward_features = self.forward_features

        # 224/4 = 32
        self.freq_dropper = DropStripes(dim=2, drop_width=32, 
            stripes_num=2)

        # # 64/8 = 8
        self.time_dropper = DropStripes(dim=3, drop_width=16, 
            stripes_num=2)
        
        # self.fc_block0 = nn.Sequential(
        #     nn.Linear(1280, 1280, bias=False),
        #     nn.BatchNorm1d(1280),
        #     nn.SiLU(),
        #     nn.Linear(1280, 1280, bias=False),
        #     )

        self.adacos = AdaCos(num_features=1280,
                             num_classes=n_classes)
        #self.out_fc = nn.Linear(1280, n_classes)
        self.forcal_loss = FocalLoss()

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

    def mixup(self, data, label, alpha=0.4, debug=False, n_classes=6, device='cuda:0'):
        #data = data.to('cpu').detach().numpy().copy()
        #label = label.to('cpu').detach().numpy().copy()
        
        batch_size = len(data)
        weights = torch.from_numpy(np.random.uniform(low=0, high=alpha, size=batch_size)).to(device)
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index, :]
        y1, y2 = label, label[index]
        x = [x1[i,:]*weights[i] + x2[i,:]*(1 - weights[i]) for i in range(batch_size)]
        x = torch.stack(x)
        y = [y1[i,:]*weights[i] + y2[i,:]*(1 - weights[i]) for i in range(batch_size)]
        y = torch.stack(y)
        return x, y

    def mixture_mask(self, data, label, alpha=0.4, n_crop_freqs_high=64, n_classes=6, device='cuda:0'):
        batch_size = data.shape[0]
        total_freq = data.shape[2]
        weights = torch.from_numpy(np.random.uniform(low=0, high=alpha, size=batch_size)).to(device)
        # random pair
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index]
        y1, y2 = label, label[index]
        # random freq range
        n_crop_freqs = torch.randint(low=1, high=n_crop_freqs_high, size=(batch_size,))
        bgn_freq = [torch.randint(low=0, high=total_freq - n_crop_freqs[i], size=(1,))[0] for i in range(batch_size)]
        bgn_freq = torch.stack(bgn_freq)
        # subst
        for i in range(batch_size):
            x1[i, :, bgn_freq[i]: bgn_freq[i]+n_crop_freqs[i], :] = x2[i, :, bgn_freq[i]: bgn_freq[i]+n_crop_freqs[i], :]
        y = [y1[i,:]*weights[i] + y2[i,:]*(1 - weights[i]) for i in range(batch_size)]
        y = torch.stack(y)
        return x1, y

    # https://github.com/google-research/augmix/blob/master/augment_and_mix.py
    def augment_and_mix(self, image, y, severity=3, width=3, depth=-1, alpha=1.):
        """Perform AugMix augmentations and compute mixture.
        Args:
            image: Raw input image as float32 np.ndarray of shape (h, w, c)
            severity: Severity of underlying augmentation operators (between 1 to 10).
            width: Width of augmentation chain
            depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
            from [1, 3]
            alpha: Probability coefficient for Beta and Dirichlet distributions.
        Returns:
            mixed: Augmented and mixed image.
        """
        ws = np.array(np.float32(
            np.random.dirichlet([alpha] * width)))
        ws = torch.from_numpy(ws).cuda()
        m = np.array(np.float32(np.random.beta(alpha, alpha)))
        m = torch.from_numpy(m).cuda()
        mix = torch.zeros_like(image)

        image_aug = image.clone()
        image_aug, y = self.mixup(image_aug, y)
        mix += ws[0] * image_aug
        image_aug = self.time_dropper(image_aug)
        mix += ws[1] * image_aug
        image_aug = self.freq_dropper(image_aug)
        mix += ws[2] * image_aug

        mixed = (1 - m) * image + m * mix
        return mixed, y
        
    def forward(self, x, section_label, label=None, device='cuda:0', is_aug=True, eval=False):
        batch_size = x.shape[0]

        x = x.transpose(1, 2)
        x = self.bn0(x)
        x = x.transpose(1, 2)

        # Data Aug
        if is_aug == True:
            y = F.one_hot(section_label, num_classes=self.n_classes)
            # plt.imshow(x[0,0,:,:].cpu().detach().numpy(), aspect='auto')
            # plt.show()
            x, y = self.augment_and_mix(x, y)
            # plt.imshow(x[0,0,:,:].cpu().detach().numpy(), aspect='auto')
            # plt.show()
        else:
            y = F.one_hot(section_label, num_classes=self.n_classes)
        #plt.imshow(x[0,0,:,:].cpu().detach().numpy(), aspect='auto')
        #plt.show()
        # effnet
        embedding, features = self.effnet(x)
        features = features.squeeze()
        pred_section = self.adacos(embedding, y)
        #pred_section = self.out_fc(embedding)
        
        loss = self.forcal_loss(pred_section, y)
        #pred_section = F.softmax(pred_section, dim=1)
        #features = torch.cat([features, pred_section], dim=1)

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