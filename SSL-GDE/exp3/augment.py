import numpy as np
import torch
from torch import nn

from torchlibrosa.augmentation import DropStripes

class Augment(nn.Module):
    def __init__(self):
        super(Augment, self).__init__()
        
        #self.bn0 = nn.BatchNorm2d(128)
        # 128/4 = 32
        self.freq_dropper = DropStripes(dim=2, drop_width=16, 
            stripes_num=2)

        # 64/8 = 8
        self.time_dropper = DropStripes(dim=3, drop_width=8, 
            stripes_num=2)
        #self.bn1 = nn.BatchNorm2d(64)
        
        # self.spec_augmenter = SpecAugmentation(time_drop_width=8,
        #                                     time_stripes_num=2,
        #                                     freq_drop_width=8,
        #                                     freq_stripes_num=2)
    
    def mixup(self, data, alpha=1, debug=False, weights=0.6):
        data = data.to('cpu').detach().numpy().copy()
        batch_size = len(data)
        weights = np.random.beta(alpha, alpha, batch_size)
        index = np.random.permutation(batch_size)
        x1, x2 = data, data[index]
        x = np.array([x1[i] * weights + x2[i] * (1 - weights) for i in range(batch_size)])
        #y1 = np.array(one_hot_labels).astype(np.float)
        #y2 = np.array(np.array(one_hot_labels)[index]).astype(np.float)
        #y = np.array([y1[i] * weights[i] + y2[i] * (1 - weights[i]) for i in range(len(weights))])
        if debug:
            print('Mixup weights', weights)
        return torch.from_numpy(x).clone().cuda()
    
    def forward(self, x):
        
        
        return x
    