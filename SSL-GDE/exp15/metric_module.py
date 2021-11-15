import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import math

class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def mixup_cross_entropy_loss(self, input, target, size_average=True):
        """Origin: https://github.com/moskomule/mixup.pytorch
        in PyTorch's cross entropy, targets are expected to be labels
        so to predict probabilities this loss is needed
        suppose q is the target and p is the input
        loss(p, q) = -\sum_i q_i \log p_i
        """
        assert input.size() == target.size()
        assert isinstance(input, Variable) and isinstance(target, Variable)
        input = torch.log(torch.nn.functional.softmax(input, dim=1).clamp(1e-5, 1))
        # input = input - torch.log(torch.sum(torch.exp(input), dim=1)).view(-1, 1)
        loss = - torch.sum(input * target)
        return loss / input.size()[0] if size_average else loss
    
    def forward(self, input, target):
        logp = self.mixup_cross_entropy_loss(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = label
        #one_hot = torch.zeros_like(logits)
        #one_hot.scatter_(1, label.view(-1, 1).long(), 1)
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