############################################################################
# load library
############################################################################

# python default library
import os
import random
import datetime
import math
import gc

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
import preprocessing_crop as prep
import preprocessing as prep_eval

from augment import Augment

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
def make_dataloader(train_paths, machine_type):   
    transform_tr = transforms.Compose([
        prep.extract_melspectrogram(eval=False)
    ])
    transform_eval = transforms.Compose([
        prep.extract_melspectrogram(eval=True)
    ])
    train_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['train'], transform=transform_tr)
    valid_source_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['valid_source'], transform=transform_eval)
    valid_target_dataset = prep.DCASE_task2_Dataset(train_paths[machine_type]['valid_target'], transform=transform_eval)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['param']['batch_size'],
        shuffle=config['param']['shuffle'],
        )
    
    valid_source_loader = torch.utils.data.DataLoader(
        dataset=valid_source_dataset,
        batch_size=1,  # 1バッチにつき一つのスペクトログラム
        shuffle=False,
        )
    
    valid_target_loader = torch.utils.data.DataLoader(
        dataset=valid_target_dataset,
        batch_size=1,
        shuffle=False,
        )

    dataloaders_dict = {"train": train_loader, "valid_source": valid_source_loader, "valid_target": valid_target_loader}
    
    return dataloaders_dict

def mixup(data, label, alpha=1, debug=False, weights=0.6, n_classes=6, device='cuda:0'):
    #data = data.to('cpu').detach().numpy().copy()
    #label = label.to('cpu').detach().numpy().copy()
    batch_size = len(data)
    label_mat = torch.zeros((batch_size, n_classes, n_classes))    # (N, C_n, C_n)
    index = np.random.permutation(batch_size)
    x1, x2 = data, data[index]
    y1, y2 = label, label[index]
    x = torch.cat([
        torch.unsqueeze(
            x1[i,:,:,:]*weights + x2[i,:,:,:]*(1 - weights),
            0) \
            for i in range(batch_size)],
            dim=0)
    # onehot 2d matrix (batch, 6, 6) => onehot vector (batch, 36) => index vector (batch, 1)
    for i in range(batch_size):
        label_mat[i, y1[i], y2[i]] = 1  # onehot
    # (classes: 0~35)
    label = torch.flatten(label_mat, start_dim=1, end_dim=-1).argmax(dim=1)
    
    return x, label

def label_transform(label):
    batch_size, n_classes = label.shape[0], 6
    label_mat = torch.zeros((batch_size, n_classes, n_classes)).cuda()
    for i in range(batch_size):
        label_mat[i, label[i], label[i]] = 1  # onehot 
    # (classes: 0~35)
    label = torch.flatten(label_mat, start_dim=1, end_dim=-1).argmax(dim=1)
    return label

def replace_label(self, labels, outlier_num=99):
    # 7の倍数を置き換え（もっと適切な方法があるはず）
    # 他の値を99で置き換え
    #labels = labels.to('cpu').detach().clone()
    labels = torch.where((labels % 7) == 0 , labels, outlier_num)
    for i in range(len(self.center_label)):
        labels[labels == self.center_label[i]] = i

    return labels

#############################################################################
# training
#############################################################################
def train_fn(data_loader, model, optimizer, epoch, device):
    # tmp_deep_feat = []
    # def hook(module, input, output):
    #     #print(output.shape)
    #     output = F.adaptive_avg_pool2d(output, 1).squeeze()
    #     tmp_deep_feat.append(output)
    # # # M7:block[5], M8:block[6], M9:act2
    # model.effnet.blocks[5].register_forward_hook(hook)
    # model.effnet.blocks[6].register_forward_hook(hook)
    # model.effnet.act2.register_forward_hook(hook)

    model.train()
    #aug = Augment()
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
    embedding_feats = []
    section_labels = []
    for iter, sample in tqdm(enumerate(data_loader), total=len(data_loader)):
        # expand
        feature = sample['feature'].to(device)
        section_label = sample['section_label'].to(device)
        #feature, section_label = mixup(feature, section_label)
        # effnet forward
        loss, embedding, _ = model.forward(feature, section_label)
        # hook
        #deep_feat = torch.cat(tmp_deep_feat, dim=1)
        # centernet forward
        #cl_loss, _ = model.forward_centerloss(deep_feat, section_label)
        #pred = F.softmax(pred, dim=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # append for output
        embedding_feats.append(embedding.to('cpu'))
        section_labels.append(section_label.to('cpu'))

        output_dict['loss'] = output_dict['loss'] + loss.to('cpu').item()
        #output_dict['feature'].append(embedding_feat.to('cpu'))
        output_dict['label'].append(sample['label'])
        output_dict['section_label'].append(sample['section_label'])
        output_dict['domain_label'].append(sample['domain_label'])
        output_dict['wav_name'].extend(sample['wav_name'])
        #output_dict['pred'].append(pred.to('cpu'))
    
    # GDE
    embedding_feats = torch.cat(embedding_feats)
    section_labels = torch.cat(section_labels)
    model.gaussian_density_estimation.set_param(embedding_feats, section_labels)
    # concat for output
    output_dict['loss'] = output_dict['loss'] / len(data_loader)
    #output_dict['feature'] = torch.cat(output_dict['feature'], dim=0).detach().numpy().copy()
    #output_dict['label'] = torch.cat(output_dict['label'], dim=0).detach().numpy().copy()
    #output_dict['section_label'] = torch.cat(output_dict['section_label'], dim=0).detach().numpy().copy()
    #output_dict['domain_label'] = torch.cat(output_dict['domain_label'], dim=0).detach().numpy().copy()
    #output_dict['pred'] = torch.cat(output_dict['pred'], dim=0).detach().numpy().copy()

    return output_dict

def validate_fn(data_loader, model, device, get_anomaly_score=False):
    # tmp_deep_feat = []
    # def hook(module, input, output):
    #     #print(output.shape)
    #     with torch.no_grad():
    #         output = F.adaptive_avg_pool2d(output, 1).squeeze()
    #         tmp_deep_feat.append(output)
    # # # M7:block[5], M8:block[6], M9:act2
    # model.effnet.blocks[5].register_forward_hook(hook)
    # model.effnet.blocks[6].register_forward_hook(hook)
    # model.effnet.act2.register_forward_hook(hook)

    model.eval()
    # init
    output_dict = {
        'loss': 0,
        'feature': [],
        'label': [],
        'section_label': [],
        'domain_label': [],
        'wav_name': [],
        'pred': [],
        'anomaly_scores': [],
        }
    # training roop
    for iter, sample in tqdm(enumerate(data_loader), total=len(data_loader)):
        # expand
        feature = sample['feature'].squeeze(0).to(device)
        size = feature.shape[0]
        section_label = torch.full(
            size=(size,),
            fill_value=sample['section_label'].item(),
            ).to(device)
        label = torch.full(
            size=(size,),
            fill_value=sample['label'].item(),
            ).to(device)
        #feature, section_label = mixup(feature, section_label)
        with torch.no_grad():
            # forward
            loss, _, pred  = model.forward(feature, section_label, label=label)
            pred = pred.mean()  # スペクトログラム一つ分のanomaly score

        # append for output
        output_dict['loss'] = output_dict['loss'] + 0
        output_dict['label'].append(sample['label'][0])
        output_dict['section_label'].append(sample['section_label'][0])
        output_dict['domain_label'].append(sample['domain_label'][0])
        output_dict['wav_name'].append(sample['wav_name'][0])
        output_dict['pred'].append(pred)
    
    # concat for output
    output_dict['loss'] = output_dict['loss']# / len(data_loader)
    output_dict['label'] = torch.stack(output_dict['label']).detach().numpy().copy()
    output_dict['section_label'] = torch.stack(output_dict['section_label']).detach().numpy().copy()
    output_dict['domain_label'] = torch.stack(output_dict['domain_label']).detach().numpy().copy()
    output_dict['pred'] = torch.stack(output_dict['pred']).to('cpu').detach().numpy().copy()
    return output_dict

# ref : https://www.kaggle.com/yasufuminakama/moa-pytorch-nn-starter
import pandas as pd
import scipy
from IPython.display import display
import dcase_util

def run_training(model, dataloaders_dict, writer, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print("use:", device)
    model.to(device)
    criterion = None
    n_epochs = config['param']['num_epochs']
    for epoch in range(n_epochs):
        output_tr = train_fn(
            data_loader=dataloaders_dict['train'],
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            )
        if epoch == n_epochs:
            output_src = validate_fn(
                data_loader=dataloaders_dict['valid_source'],
                model=model,
                device=device,
                get_anomaly_score=True
                )
            output_tgt = validate_fn(
                data_loader=dataloaders_dict['valid_target'],
                model=model,
                device=device,
                get_anomaly_score=True,
                )
        else:
            if ((epoch+1) % 10 == 0) or (epoch == 0):
                output_src = validate_fn(
                    data_loader=dataloaders_dict['valid_source'],
                    model=model,
                    device=device,
                    )
                output_tgt = validate_fn(
                    data_loader=dataloaders_dict['valid_target'],
                    model=model,
                    device=device,
                    )
        # tr
        tr_loss, src_loss, tgt_loss = output_tr['loss'], output_src['loss'], output_tgt['loss']
        tr_feat, tr_sec = output_tr['feature'], output_tr['section_label']
        # src
        src_feat, src_sec = output_src['feature'], output_src['section_label']
        src_label, src_domain = output_src['label'], output_src['domain_label']
        # tgt
        tgt_feat, tgt_sec = output_tgt['feature'], output_tgt['section_label']
        tgt_label, tgt_domain = output_tgt['label'], output_tgt['domain_label']
        
        # pred
        src_pred = output_src['pred']
        tgt_pred = output_tgt['pred']
        
        # pred df
        src_pred_df = dcase_util.make_pred_df(output_src['wav_name'], src_sec, src_domain, src_label, src_pred)
        tgt_pred_df = dcase_util.make_pred_df(output_tgt['wav_name'], tgt_sec, tgt_domain, tgt_label, tgt_pred)
        pred_df = pd.concat([src_pred_df, tgt_pred_df], axis=0)
        
        # calc score
        src_score_df = dcase_util.calc_dcase2021_task2_score(src_pred_df, prefix='Source')
        tgt_score_df = dcase_util.calc_dcase2021_task2_score(tgt_pred_df, prefix='Target')
        # per mean auc
        src_mean_auc, tgt_mean_auc = src_score_df['AUC'].mean(axis=0), tgt_score_df['AUC'].mean(axis=0)
        # concat score
        score_df = pd.concat([src_score_df, tgt_score_df], axis=0)
        mean = pd.DataFrame([score_df.mean()], index=['mean'])
        hmean = scipy.stats.hmean(score_df, axis=0)
        hmean = pd.DataFrame([hmean], columns=['AUC', 'pAUC'], index=['h_mean'])
        score_df = score_df.append([mean, hmean])
        auc_score = mean['AUC'].iloc[0]
        epoch_log = (
            f'epoch:{epoch+1}/{n_epochs},'
            f' tr_loss:{tr_loss:.6f},'
            f' src_loss:{src_loss:.6f},'
            f' src_mean_auc:{src_mean_auc:.6f},'
            f' tgt_loss:{tgt_loss:.6f},'
            f' tgt_mean_auc:{tgt_mean_auc:.6f},'
            f' mean_auc:{auc_score:.6f},'
        )
        logger.info(epoch_log)
        # show score_df
        display(score_df)
        
    output_dict = {'train':output_tr, 'val_src':output_src, 'val_tgt':output_tgt}
    return output_dict, model, pred_df, score_df