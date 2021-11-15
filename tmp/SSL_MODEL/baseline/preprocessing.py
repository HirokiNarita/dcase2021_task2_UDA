########################################################################
# import python-library
########################################################################
# python library
import sys
from numpy.lib.nanfunctions import _nanmedian_small
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np
import torch
import librosa
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

from torchlibrosa.augmentation import SpecAugmentation
#from torchaudio.transforms import Resample
# original library
import common as com
#########################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)

class extract_melspectrogram(object):
    """
    データロード(波形)
    
    Attributes
    ----------
    sound_data : logmelspectrogram
    """
    def __init__(self, sound_data=None):
        self.sample_rate=config['param']['sample_rate']
        self.n_mels = config['param']['mel_bins']
        self.n_fft = config['param']['window_size']
        self.hop_length=config['param']['hop_size']
        self.power = 2.0
        
        self.sound_data = sound_data
        self.mel_spectrogram_transformer = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
        ).cuda()
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
        #                                        freq_drop_width=8, freq_stripes_num=2)
        #self.resample = T.Resample(16000, config['param']['sample_rate']).cuda()
    
    def extract(self, file_name):
        X, _ = torchaudio.load(file_name)
        X = X.cuda()
        X = self.mel_spectrogram_transformer(X)
        eps = 1e-16
        X = (
            20.0 / self.power * torch.log10(X + eps)
        )
        X = X.cpu()
        X = X.squeeze()
        return X
    
    def make_subseq(self, X, training=True):
        # mel_spectrogram is np.ndarray [shape=(n_mels, t)]
        n_frames = config['param']['n_crop_frames']
        n_hop_frames = config['param']['n_hop_frames']
        total_frames = X.shape[1] - n_frames + 1
        subseq = []
        # generate feature vectors by concatenating multiframes
        for frame_idx in range(total_frames):
            subseq.append(X[:, frame_idx:frame_idx+n_frames])
        
        subseq = torch.stack(subseq, dim=0)
        # reduce sample
        if training:
            vectors = subseq[:: n_hop_frames, :, :]

        return vectors

def get_detail(file_name):
    # param
    label = com.get_label(file_name)
    section_label = com.get_section_type(file_name)
    domain_label = com.get_domain_label(file_name)
    
    return {
        'wav_name': file_name,
        'label': torch.from_numpy(np.array(label)),
        'section_label': torch.from_numpy(np.array(section_label)),
        'domain_label': torch.from_numpy(np.array(domain_label)),}


class DCASE_task2_Dataset(torch.utils.data.Dataset):
    '''
    Attribute
    ----------
    
    '''
    
    def __init__(self, file_list, transform=None, training=True):
        self.transform = transform
        self.file_list = file_list
        self.dataset = []
        self.ext_melspec = extract_melspectrogram()
        self.training = training
        """
        self.dataset = [sample_dict_1, sample_dict_2 , ...]
        sample_dict = {
            'feature': (var)
            'label': (var),
            'section_label': (var),
            'domain_label': (var),
            'wav_name': (var),}
        """
        
        for file_name in tqdm(file_list):
            # extract feature
            feature = self.ext_melspec.extract(file_name)                  # [shape=(n_mels, t)]
            # plt.imshow(feature[0], aspect='auto')
            # plt.show()
            feature = self.ext_melspec.make_subseq(feature, training=self.training)                # [shape=(samples, n_mels, t)]
            # details
            details_dict = get_detail(file_name)
            # make dataset
            self.n_samples = feature.shape[0]
            tmp_dataset = [details_dict for i in range(self.n_samples)]
            #print(feature.shape)
            # subst feature (copyじゃないとバグるので注意)
            for idx in range(self.n_samples):
                tmp_dataset[idx]['feature'] = feature[idx, :, :]
                self.dataset.append(tmp_dataset[idx])
            
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]  # return dict

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_per_sample_size(self):
        return self.n_samples