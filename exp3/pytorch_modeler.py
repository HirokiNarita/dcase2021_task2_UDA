############################################################################
# load library
############################################################################

# python default library
import os
import random
import datetime
import math


# general analysis tool-kit
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# pytorch
import torch
import torch.utils.data as data
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter

# deeplearning tool-kit
from torchvision import transforms

# etc
import yaml
yaml.warnings({'YAMLLoadWarning': False})
from tqdm import tqdm
from collections import defaultdict

# original library
import common as com
import preprocessing as prep

############################################################################
# load config
############################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)
log_folder = config['IO_OPTION']['OUTPUT_ROOT']+'/{0}.log'.format(datetime.date.today())
logger = com.setup_logger(log_folder, 'pytorch_modeler.py')
############################################################################
# Setting seed
############################################################################
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

############################################################################
# Make Dataloader
############################################################################
def make_dataloader(train_paths, machine_type, mode='training'):
    transform_tr = transforms.Compose([
        prep.extract_melspectrogram(mode='training')
    ])
    transform_eval = transforms.Compose([
        prep.extract_melspectrogram(mode='eval')
    ])
    train_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['train'], transform=transform_tr)
    valid_source_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['valid_source'], transform=transform_eval)
    valid_target_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['valid_target'], transform=transform_eval)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=config['param']['shuffle'],
        num_workers=2,
        pin_memory=True
        )
    
    valid_source_loader = torch.utils.data.DataLoader(
        dataset=valid_source_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
        )
    
    valid_target_loader = torch.utils.data.DataLoader(
        dataset=valid_target_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
        )

    dataloaders_dict = {"train": train_loader, "valid_source": valid_source_loader, "valid_target": valid_target_loader}
    
    return dataloaders_dict

#############################################################################
# training
#############################################################################
def calc_auc(y_true, y_pred):
    auc = metrics.roc_auc_score(y_true, y_pred)
    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=config["etc"]["max_fpr"])
    #logger.info("AUC : {}".format(auc))
    #logger.info("pAUC : {}".format(p_auc))
    return auc, p_auc

def make_subseq(X, hop_mode=False):
    
    n_mels = config['param']['mel_bins']
    n_crop_frames = config['param']['n_crop_frames']
    n_hop_frames = config['param']['extract_hop_len']
    total_frames = len(X.shape[3]) - n_crop_frames + 1
    subseq = []
    # generate feature vectors by concatenating multiframes
    for frame_idx in range(total_frames):
        subseq.append(X[:,:,frame_idx:(frame_idx+1)*n_crop_frames])
    subseq = torch.cat(subseq, dim=0)
    # reduce sample
    if hop_mode:
        vectors = subseq[:,:,:: n_hop_frames]
    
    return vectors

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
        dist = (x-center).pow(2).sum(dim=-1)
        loss = torch.clamp(dist, min=1e-12, max=1e+12).mean(dim=-1)

        return loss

def extract_model(model, dataloaders_dict):
    model.eval()
    #scaler = torch.cuda.amp.GradScaler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    model.to(device)
    for phase in ['train']:
        labels = []
        wav_names = []
        features = []
        losses = 0
        for step, sample in enumerate(tqdm(dataloaders_dict[phase])):
            wav_name = sample['wav_name']
            wav_names = wav_names + wav_name
            label = sample['label'].to('cpu')
            labels.append(label)
            feature = sample['feature']   # (batch, ch, mel_bins, n_frames)
            feature = feature.to(device)

            #with torch.cuda.amp.autocast():
            with torch.no_grad():
                feature = model(feature)
            feature = feature.to('cpu')
            features.append(feature)
    
    # processing per epoch
    features = torch.cat(features, dim=0).detach().numpy().copy()
    labels = torch.cat(labels, dim=0).detach().numpy().copy()
    # end
    output_dicts = {'features': features, 'wav_names': wav_names, 'labels': labels}
    
    return output_dicts

def train_fn(data_loader, model, optimizer, criterion, scaler, device):
    model.train()
    # hock
    # def hook(module, input, output):
    #     #print(output.shape)
    #     global mid_feat
    #     mid_feat = output.cpu()
    #model.effnet.global_pool.register_forward_hook(hook)
    # init
    output_dict = {
        'loss': 0,
        'feature': [],
        'label': [],
        'section_label': [],
        'domain_label': [],
        'wav_name': [],
        'pred': [],
        }
    # training roop
    for iter, sample in enumerate(tqdm(data_loader)):
        # expand
        feature = sample['feature'].to(device)
        section_label = sample['section_label'].to(device)
        # propagation
        #optimizer.zero_grad()
        #with torch.cuda.amp.autocast():
        pred, embedding_feat = model(feature)
        loss = criterion(pred, section_label)
        pred = F.softmax(pred, dim=1)
        #scaler.scale(loss).backward()
        #scaler.step(optimizer)
        #scaler.update()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # append for output
        output_dict['loss'] = output_dict['loss'] + loss.item()
        output_dict['feature'].append(embedding_feat.to('cpu'))
        output_dict['label'].append(sample['label'])
        output_dict['section_label'].append(sample['section_label'])
        output_dict['domain_label'].append(sample['domain_label'])
        output_dict['wav_name'].extend(sample['wav_name'])
        output_dict['pred'].append(pred.to('cpu'))
    # concat for output
    output_dict['feature'] = torch.cat(output_dict['feature'], dim=0).detach().numpy().copy()
    output_dict['label'] = torch.cat(output_dict['label'], dim=0).detach().numpy().copy()
    output_dict['section_label'] = torch.cat(output_dict['section_label'], dim=0).detach().numpy().copy()
    output_dict['domain_label'] = torch.cat(output_dict['domain_label'], dim=0).detach().numpy().copy()
    output_dict['pred'] = torch.cat(output_dict['pred'], dim=0).detach().numpy().copy()
    
    return output_dict

def validate_fn(data_loader, model, criterion, device):
    model.eval()
    # hock
    # def hook(module, input, output):
    #     #print(output.shape)
    #     global mid_feat
    #     mid_feat = output.cpu()
    # model.effnet.global_pool.register_forward_hook(hook)
    # init
    output_dict = {
        'loss': 0,
        'feature': [],
        'label': [],
        'section_label': [],
        'domain_label': [],
        'wav_name': [],
        'pred': []
        }
    # training roop
    for iter, sample in enumerate(tqdm(data_loader)):
        # expand
        feature = sample['feature'].to(device)
        section_label = sample['section_label'].to(device)
        # propagation
        #with torch.cuda.amp.autocast():
        with torch.no_grad():
            pred, embedding_feat = model(feature)
            loss = criterion(pred, section_label)
            pred = F.softmax(pred, dim=1)
                #pred = torch.max(pred.data, 1)
        # append for output
        output_dict['loss'] = output_dict['loss'] + loss.item()
        output_dict['feature'].append(embedding_feat.to('cpu'))
        output_dict['label'].append(sample['label'])
        output_dict['section_label'].append(sample['section_label'])
        output_dict['domain_label'].append(sample['domain_label'])
        output_dict['wav_name'].extend(sample['wav_name'])
        output_dict['pred'].append(pred.to('cpu'))
    # concat for output
    output_dict['feature'] = torch.cat(output_dict['feature'], dim=0).detach().numpy().copy()
    output_dict['label'] = torch.cat(output_dict['label'], dim=0).detach().numpy().copy()
    output_dict['section_label'] = torch.cat(output_dict['section_label'], dim=0).detach().numpy().copy()
    output_dict['domain_label'] = torch.cat(output_dict['domain_label'], dim=0).detach().numpy().copy()
    output_dict['pred'] = torch.cat(output_dict['pred'], dim=0).detach().numpy().copy()
    
    return output_dict
 
# ref : https://www.kaggle.com/yasufuminakama/moa-pytorch-nn-starter
def run_training(model, dataloaders_dict, writer, optimizer):
    #scaler = torch.cuda.amp.GradScaler()
    scaler = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    n_epochs = config['param']['num_epochs']
    for epoch in range(n_epochs):
        output_tr = train_fn(
            data_loader=dataloaders_dict['train'],
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            )
        output_src = validate_fn(
            data_loader=dataloaders_dict['valid_source'],
            model=model,
            criterion=criterion,
            device=device,
            )
        output_tgt = validate_fn(
            data_loader=dataloaders_dict['valid_target'],
            model=model,
            criterion=criterion,
            device=device,
            )
        
        tr_loss, src_loss, tgt_loss = output_tr['loss'], output_src['loss'], output_tgt['loss']
        src_pred, src_label = output_src['pred'], output_src['section_label']
        tgt_pred, tgt_label = output_tgt['pred'], output_tgt['section_label']
        
        src_acc = metrics.accuracy_score(src_label, np.argmax(src_pred, axis=1))
        tgt_acc = metrics.accuracy_score(tgt_label, np.argmax(tgt_pred, axis=1))
        epoch_log = (
            f'epoch:{epoch+1}/{n_epochs},'
            f' tr_loss:{tr_loss:.6f},'
            f' src_loss:{src_loss:.6f},'
            f' src_acc:{src_acc:.6f},'
            f' tgt_loss:{tgt_loss:.6f},'
            f' tgt_acc:{tgt_acc:.6f},'
        )
        logger.info(epoch_log)
    output_dict = {'train':output_tr, 'val_src':output_src, 'val_tgt':output_tgt}
    return output_dict, model