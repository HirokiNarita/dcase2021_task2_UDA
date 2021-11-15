########################################################################
# import python-library
########################################################################
# python library
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np
import torch
import librosa
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt

from torchlibrosa.augmentation import SpecAugmentation
#from torchaudio.transforms import Resample
# original library
import common as com
#########################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)

def random_crop(X):
    n_crop_frames = config['param']['n_crop_frames']
    total_frames = X.shape[1]
    bgn_frame = torch.randint(low=0, high=total_frames - n_crop_frames, size=(1,))[0]
    X = X[:, bgn_frame: bgn_frame+n_crop_frames]
    return X

def make_subseq(X, hop_mode=False):
    # mel_spectrogram is np.ndarray [shape=(n_mels, t)]
    n_frames = config['param']['n_crop_frames']
    #n_hop_frames = config['param']['n_hop_frames']
    total_frames = X.shape[1] - n_frames + 1
    subseq = []
    # generate feature vectors by concatenating multiframes
    for frame_idx in range(total_frames):
        subseq.append(X[:, frame_idx:frame_idx+n_frames])
    
    subseq = torch.stack(subseq, dim=0)
    # reduce sample
    return subseq

class extract_melspectrogram(object):
    """
    データロード(波形)
    
    Attributes
    ----------
    sound_data : logmelspectrogram
    """
    def __init__(self, sound_data=None, eval=False):
        self.sample_rate=config['param']['sample_rate']
        self.n_mels = config['param']['mel_bins']
        self.n_fft = config['param']['window_size']
        self.hop_length=config['param']['hop_size']
        self.power = 2.0
        
        self.sound_data = sound_data
        self.eval = eval
        self.mel_spectrogram_transformer = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
        )
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
        #                                        freq_drop_width=8, freq_stripes_num=2)
    
    def __call__(self, sample):
        if self.eval == True:
            X, _ = torchaudio.load(sample['wav_name'])
            X = X.squeeze()
            X = self.mel_spectrogram_transformer(X)
            eps = 1e-16
            X = (
                20.0 / self.power * torch.log10(X + eps)
            )
            X = make_subseq(X)   # [n_samples, n_mels, t]
            X = torch.stack([X,X,X], dim=1)
            self.sound_data = X
            self.wav_name = sample['wav_name']
        else:
            #source
            X_src, _ = torchaudio.load(sample['wav_name']['source'])
            X_src = X_src.squeeze()
            X_src = self.mel_spectrogram_transformer(X_src)
            eps = 1e-16
            X_src = (
                20.0 / self.power * torch.log10(X_src + eps)
            )
            X_src = random_crop(X_src)   # [n_mels, t]
            X_src = torch.stack([X_src,X_src,X_src], dim=0) # [ch, n_mels, t]
            #tgt
            tgt_wavnames = sample['wav_name']['target']
            tgt_idx = torch.randint(low=0, high=len(tgt_wavnames)-1, size=(1,))[0]
            X_tgt, _ = torchaudio.load(sample['wav_name']['target'][tgt_idx])
            X_tgt = X_tgt.squeeze()
            X_tgt = self.mel_spectrogram_transformer(X_tgt)
            eps = 1e-16
            X_tgt = (
                20.0 / self.power * torch.log10(X_tgt + eps)
            )
            X_tgt = random_crop(X_tgt)   # [n_mels, t]
            X_tgt = torch.stack([X_tgt,X_tgt,X_tgt], dim=0) # [ch, n_mels, t]
            self.sound_data = [X_src, X_tgt]
            self.wav_name = sample['wav_name']['source']

        ############################
        self.label = np.array(sample['label'])
        self.section_label = np.array(com.get_section_type(self.wav_name))
        self.domain_label = np.array(com.get_domain_label(self.wav_name))
        
        return {'feature': self.sound_data,
                'label': torch.from_numpy(self.label),
                'section_label': torch.from_numpy(self.section_label),
                'domain_label': torch.from_numpy(self.domain_label),
                'wav_name': self.wav_name}

class DCASE_task2_Dataset(torch.utils.data.Dataset):
    '''
    Attribute
    ----------
    
    '''
    
    def __init__(self, file_list, transform=None, eval=False):
        self.transform = transform
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # ファイル名でlabelを判断
        if "normal" in file_path:
            label = 0
        else:
            label = 1
        
        sample = {'wav_name':file_path, 'label':np.array(label)}
        sample = self.transform(sample)
        
        return sample
    
    # def get_per_sample_size(self):
    #     return self.n_samples
