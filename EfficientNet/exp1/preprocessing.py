########################################################################
# import python-library
########################################################################
# python library
import sys
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np
import torch
import librosa
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchlibrosa.augmentation import SpecAugmentation
#from torchaudio.transforms import Resample
# original library
import common as com
#########################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)

def make_log_mel_spectrogram(file_name):
    # param
    sample_rate=config['param']['sample_rate']
    n_mels = config['param']['mel_bins']
    n_fft = config['param']['window_size']
    hop_length=config['param']['hop_size']
    power = 2.0
    
    # generate melspectrogram using librosa
    audio, sr = librosa.load(file_name,
                             sr=16000,
                             mono=True)
    # mel_spectrogram is np.ndarray [shape=(n_mels, t)]
    # n_mels is the number of mel-filterbanks, t is the number of frames
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
    )
    # convert melspectrogram to log mel energies
    log_mel_spectrogram = (
        20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))
    )
    return log_mel_spectrogram

def make_subseq(X, hop_mode=False):
    # mel_spectrogram is np.ndarray [shape=(n_mels, t)]
    n_frames = config['param']['n_frames']
    #n_hop_frames = config['param']['n_hop_frames']
    total_frames = X.shape[1] - n_frames + 1
    subseq = []
    # generate feature vectors by concatenating multiframes
    for frame_idx in range(total_frames):
        subseq.append(X[:, frame_idx:frame_idx+n_frames])
    
    subseq = np.stack(subseq, axis=0)
    # reduce sample
    return subseq


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
    
    def __init__(self, file_list, transform=None):
        self.transform = transform
        self.file_list = file_list
        self.dataset = []
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
            feature = make_log_mel_spectrogram(file_name) # [shape=(n_mels, t)]
            feature = make_subseq(feature)                # [shape=(samples, n_mels, t)]
            feature = feature.astype(np.float32)
            # details
            details_dict = get_detail(file_name)
            # make dataset
            self.n_samples = feature.shape[0]
            tmp_dataset = [details_dict] * self.n_samples
            # subst feature (copyじゃないとバグるので注意)
            for idx in range(self.n_samples):
                tmp_dataset[idx]['feature'] = torch.from_numpy(feature[idx, :, :])
                self.dataset.append(tmp_dataset[idx].copy())
            
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]  # return dict

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_per_sample_size(self):
        return self.n_samples