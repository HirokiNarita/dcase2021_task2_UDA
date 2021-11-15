import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import faiss
#import ResNet

mvtype = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
          'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
          'wood', 'zipper']

transform_color = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_gray = transforms.Compose([
                                 transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

# def get_resnet_model(resnet_type=152):
#     """
#     A function that returns the required pre-trained resnet model
#     :param resnet_number: the resnet type
#     :return: the pre-trained model
#     """
#     if resnet_type == 18:
#         return ResNet.resnet18(pretrained=True, progress=True)
#     elif resnet_type == 50:
#         return ResNet.wide_resnet50_2(pretrained=True, progress=True)
#     elif resnet_type == 101:
#         return ResNet.resnet101(pretrained=True, progress=True)
#     else:  #152
#         return ResNet.resnet152(pretrained=True, progress=True)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def freeze_parameters(model):
    """
    1~9層のうち，
    7,8,9層以外をfreeze

    Args:
        model ([type]): [description]
        train_fc (bool, optional): [description]. Defaults to False.
    """
    # bn0
    for p in model.bn0.parameters():
        p.requires_grad = False
    # block 1
    for p in model.effnet.conv_stem.parameters():
        p.requires_grad = False
    for p in model.effnet.bn1.parameters():
        p.requires_grad = False
    # block 2~8 (model.effnet.blocks[i] -> 7 blocks)
    for i in range(3):
        for p in model.effnet.blocks[i].parameters():
            p.requires_grad = False
    # block 9
    # for p in model.effnet.conv_head.parameters():
    #     p.requires_grad = True
    # for p in model.effnet.bn2.parameters():
    #     p.requires_grad = True

def knn_score(train_set, test_set, n_neighbours=2):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D, axis=1)

def get_outliers_loader(batch_size):
    dataset = torchvision.datasets.ImageFolder(root='./data/tiny', transform=transform_color)
    outlier_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return outlier_loader

def get_loaders(dataset, label_class, batch_size):
    if dataset in ['cifar10', 'fashion']:
        if dataset == "cifar10":
            ds = torchvision.datasets.CIFAR10
            transform = transform_color
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)
        elif dataset == "fashion":
            ds = torchvision.datasets.FashionMNIST
            transform = transform_gray
            coarse = {}
            trainset = ds(root='data', train=True, download=True, transform=transform, **coarse)
            testset = ds(root='data', train=False, download=True, transform=transform, **coarse)

        idx = np.array(trainset.targets) == label_class
        testset.targets = [int(t != label_class) for t in testset.targets]
        trainset.data = trainset.data[idx]
        trainset.targets = [trainset.targets[i] for i, flag in enumerate(idx, 0) if flag]
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
        return train_loader, test_loader
    else:
        print('Unsupported Dataset')
        exit()

def clip_gradient(optimizer, grad_clip):
    assert grad_clip>0, 'gradient clip value must be greater than 1'
    for group in optimizer.param_groups:
        for param in group['params']:
            # gradient
            if param.grad is None:
                continue
            param.grad.data.clamp_(-grad_clip, grad_clip)

