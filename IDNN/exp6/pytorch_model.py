import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torchaudio.transforms import Resample

import matplotlib.pyplot as plt

class CenterLoss(nn.Module):
    def __init__(self, num_class=10, num_feature=2):
        super(CenterLoss, self).__init__()
        self.num_class = num_class
        self.num_feature = num_feature
        self.centers = nn.Parameter(torch.randn(self.num_class, self.num_feature))

    def forward(self, x, labels=None):
        if labels == None:
            labels = torch.zeros(x.shape[0]).long().cuda()
        center = self.centers[labels]
        dist = (x-center).pow(2)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss

from torch.autograd import Function
class GradientReversalLayer(Function):
    @staticmethod
    def forward(context, x, constant):
        context.constant = constant
        return x.view_as(x) * constant

    @staticmethod
    def backward(context, grad):
        return grad.neg() * context.constant, None

class FC_block(nn.Module):
    def __init__(self, in_features, out_features):
        
        super(FC_block, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(out_features)
        #self.ln1 = nn.LayerNorm(out_features)
    
    def forward(self, input, ):
        x = input
        x = self.relu(self.bn1(self.fc1(x)))
        return x

class IDNN(nn.Module):
    def __init__(self, in_size=1252, out_size=313, classes=6):
        super(IDNN, self).__init__()
        #モデルの定義
        self.rnn = nn.LSTM(input_size=4, hidden_size=4, num_layers=2, batch_first=True)
        
        # F (encoder)
        # self.enc_block1 = FC_block(5, 1)
        # self.enc_block2 = FC_block(128, 64)
        # self.enc_block3 = FC_block(64, 32)
        
        # # D (domain adversarial)
        # # self.discriminator_block1 = FC_block(32, 32)
        # # self.discriminator_block2 = FC_block(32, 1)
        # # self.adv_criterion = nn.MSELoss()
        
        # # C (predictor)
        # self.predictor_block1 = FC_block(32, 64)
        self.predictor_block2 = FC_block(4, 4)
        self.predictor_block3 = nn.Linear(4, 1)

   
    def forward(self, X, section_label, alpha=1.0):
        output = {}
        # [X=(samples, n_mels, t)]
        # [x=(samples, n_mels, 4)]
        x = torch.cat([X[:, 0:2, :], X[:, 3:5, :]], dim=1).transpose(1,2)
        # flatten [x=(samples, n_mels*4)]
        #x = torch.flatten(x, start_dim=1, end_dim=2)
        # [x=(samples, n_mels)]
        y = X[:, 2, :].unsqueeze(1).transpose(1,2)
        # network
        # F (encoder)
        #print(x.shape)
        #print(y.shape)
        x, _ = self.rnn(x)
        #print(x.shape)
        #x = self.enc_block2(x)
        #z = self.enc_block3(x)
        
        # if self.training == True:
        #     # D (domain adversarial)
        #     y_D = GradientReversalLayer.apply(z, alpha)
        #     y_D = self.discriminator_block1(y_D)
        #     y_D = self.discriminator_block2(y_D)
        #     adv_loss = self.adv_criterion(y_D.squeeze(), section_label.float())
        #     output['adv_loss'] = adv_loss
            
        # C (predictor)
        #y_C = self.predictor_block1(z)
        #y_C = self.predictor_block2(y_C)
        y_C = self.predictor_block3(x)
        
        # anomaly score
        anomaly_score = F.mse_loss(y_C, y, reduction='none')
        output['pred'] = anomaly_score
        return output