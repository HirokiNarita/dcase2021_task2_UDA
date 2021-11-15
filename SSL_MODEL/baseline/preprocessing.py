########################################################################
# import python-library
########################################################################
# python library
import yaml
yaml.warnings({'YAMLLoadWarning': False})
import numpy as np
import torch
import torchaudio
import librosa
import torchaudio.transforms as T
import matplotlib.pyplot as plt

from torchlibrosa.augmentation import SpecAugmentation
#from torchaudio.transforms import Resample
# original library
import common as com
#########################################################################
with open("./config.yaml", 'rb') as f:
    config = yaml.load(f)

# def random_crop(X):
#     n_crop_frames = config['param']['n_crop_frames']
#     total_frames = X.shape[1]
#     bgn_frame = torch.randint(low=0, high=total_frames - n_crop_frames, size=(1,))[0]
#     X = X[:, bgn_frame: bgn_frame+n_crop_frames]
#     return X

# def make_subseq(X, hop_mode=False):
#     # mel_spectrogram is np.ndarray [shape=(n_mels, t)]
#     n_frames = config['param']['n_crop_frames']
#     #n_hop_frames = config['param']['n_hop_frames']
#     total_frames = X.shape[1] - n_frames + 1
#     subseq = []
#     # generate feature vectors by concatenating multiframes
#     for frame_idx in range(total_frames):
#         subseq.append(X[:, frame_idx:frame_idx+n_frames])
    
#     subseq = torch.stack(subseq, dim=0)
#     # reduce sample
#     return subseq

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
        self.img_size = 256
        
        self.sound_data = sound_data
        self.eval = eval
        self.mel_spectrogram_transformer = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=self.power,
        ).cuda()
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
        #                                        freq_drop_width=8, freq_stripes_num=2)
    
    def __call__(self, sample):
        X, _ = torchaudio.load(sample['wav_name'])
        X = X.squeeze()
        X = X.cuda()
        X = self.mel_spectrogram_transformer(X)
        eps = 1e-16
        X = (
            20.0 / self.power * torch.log10(X + eps)
        )
        #X = X.cpu()
        X = torch.stack([X,X,X], dim=0) # [ch, n_mels, t]
        # ##### for imagenet model####
        # X = X.cpu().squeeze()
        # X = X.detach().numpy().copy()
        # X = self.mono_to_color(X)
        # X = cv2.resize(X, (self.img_size, self.img_size))
        # X = np.moveaxis(X, 2, 0)
        # X = (X / 255.0).astype(np.float32)
        # X = torch.from_numpy(X.astype(np.float32)).clone()
        # ############################
        self.sound_data = X
        self.label = np.array(sample['label'])
        self.wav_name = sample['wav_name']
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
