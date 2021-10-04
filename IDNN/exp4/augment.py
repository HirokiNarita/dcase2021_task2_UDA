import numpy as np
import torch
from torch import nn

from torchlibrosa.augmentation import SpecAugmentation

class Augment(nn.Module):
    def __init__(self):
        super(Augment, self).__init__()
        
        #self.bn0 = nn.BatchNorm2d(64)
        #self.bn1 = nn.BatchNorm2d(64)
        
        # self.spec_augmenter = SpecAugmentation(time_drop_width=8,
        #                                     time_stripes_num=2,
        #                                     freq_drop_width=8,
        #                                     freq_stripes_num=2)
    
    def mixup(self, data, alpha=1, debug=False, weights=0.4):
        data = data.to('cpu').detach().numpy().copy()
        batch_size = len(data)
        #weights = np.random.beta(alpha, alpha, batch_size)
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index]
        x = np.array([x1[i] * weights + x2[i] * (1 - weights) for i in range(batch_size)])
        #x = np.array([x1[i] * weights[i] + x2[i] * (1 - weights[i]) for i in range(len(weights))])
        if debug:
            print('Mixup weights', weights)
        return torch.from_numpy(x).clone()
    
    def forward(self, x):
        #x = x.transpose(1,3)
        #x = self.bn0(x)
        #x = x.transpose(1,3)
        x = self.mixup(x)
        #x = self.spec_augmenter(x)
        #x = x.transpose(1,3)
        #x = self.bn1(x)
        #x = x.transpose(1,3)
        
        return x
    