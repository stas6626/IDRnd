import random

import numpy as np
import librosa
import torch
from sonopy import mfcc_spec

class ToMono:
    def __call__(self, audio):
        return audio.mean(1, dtype=int)

class Normalize:
    def __call__(self, audio):
        if np.std(audio) == 0:
            return audio - np.mean(audio)
        else:
            return (audio - np.mean(audio)) / np.std(audio)
        
class Normalize_predef:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, audio):
        return (audio - self.mean) / self.std

class RandomNoise:
    def __init__(self, std, p=0.2):
        self.std = std
        self.p = p
    
    def __call__(self, audio):
        if random.random()<self.p:
        #noise = numpy.random.normal(0, self.std, len(audio))
            return audio+np.random.normal(0, self.std, len(audio))
        else:
            return audio

class Shift:
    def __init__(self, shift, p=0.2):
        self.shift = int(shift)
        self.p = p

    def __call__(self, audio):
        if random.random()<self.p:
            return np.roll(audio, self.shift)
        else:
            return audio
        
class TimeStretch:
    def __init__(self, rate, p=0.2):
        self.rate = rate
        self.p = p

    def __call__(self, audio):
        if random.random()<self.p:
            return librosa.effects.time_stretch(audio, self.rate)
        else:
            return audio

class Distortion:
    def __init__(self, min_=-5000, max_=5000, p=0.2):
        self.min_ = min_
        self.max_ = max_
        self.p = p

    def __call__(self, audio):
        if random.random()<self.p:
            return np.clip(audio, self.min_, self.max_)
        else:
            return audio

class GetMFCC:
    def __init__(self):
        self.sampling_rate = 44100
        self.window_stride = (160,80)
        self.fft_size = 512
        self.num_filt = 20
        self.num_coeffs = 13
        #self.max_frame_len = 16000

    def __call__(self, audio):
            mfccs = mfcc_spec(audio, self.sampling_rate, window_stride=self.window_stride,
                    fft_size=self.fft_size, num_filt=self.num_filt, num_coeffs=self.num_coeffs)
            #mfccs = self.normalize(mfccs)
            #diff = self.max_frame_len - mfccs.shape[0]
            #mfccs = np.pad(mfccs, ((0, diff), (0, 0)), "constant")
            return mfccs.T

class RandomParameter:
    def __init__(self, aug, parametres:list, p=0.2, num_params=1):
        self.aug = aug
        self.parametres = parametres
        self.p = p
        self.num_params = num_params

    def __call__(self, audio):
        if random.random()<self.p:
            
            params = []
            for min_, max_ in self.parametres:
                params.append(random.uniform(min_, max_))

            aug = self.aug(*params, p=1)
            return aug(audio)
        
        else:
            return audio

class PitchShift:
    def __init__(self, n_steps, p=0.2):
        self.n_steps = int(n_steps)
        self.p = p
        self.sr = 44100

    def __call__(self, audio):
        if random.random()<self.p:
            return librosa.effects.pitch_shift(audio, self.sr, self.n_steps)
        else:
            return audio

class MFCCToTensor:
    def __call__(self, mfcc):
        return torch.Tensor(mfcc).float().transpose(0, 1)

class ToMellSpec:
    def __init__(self, n_fft=2048, hop_length=512, n_mels=128):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def __call__(self, audio):
        mel = librosa.feature.melspectrogram(audio, sr=44100, n_mels=self.n_mels,
                                             n_fft=self.n_fft, hop_length=self.hop_length)
        return mel

class PadOrClip:
    def __init__(self, pad_lenth):
        self.pad_lenth = pad_lenth
    
    def __call__(self, mel):
        if len(mel.shape)==2:
            if mel.shape[1] < self.pad_lenth:
                mel = np.pad(mel, [(0, 0), (0, self.pad_lenth-mel.shape[1])], "constant")
            else:
                right_limit = mel.shape[1] - self.pad_lenth
                shift = random.randint(0, right_limit)

                mel = mel[:, shift: shift+self.pad_lenth]
            return mel
        
        elif len(mel.shape)==1:
            if mel.shape[0] < self.pad_lenth:
                mel = np.pad(mel, [0, self.pad_lenth-mel.shape[0]], "constant")
            else:
                right_limit = mel.shape[0] - self.pad_lenth
                shift = random.randint(0, right_limit)

                mel = mel[shift: shift+self.pad_lenth]
            return mel

class VTLP:
    def __init__(self, alpha=1.2, p=1):
        self.alpha = alpha
        self.sr = 44100
        self.f_hi = 0.15
    
    def __call__(self, audio):
        min_ = min(self.alpha, 1)
        new_audio = np.where(np.abs(audio) <= self.f_hi * (min_/self.alpha), audio*self.alpha, self.sr/2 - (self.sr/2 - self.f_hi*min_)/(self.sr/2 - self.f_hi*(min_/self.alpha))*(self.sr/2-audio))
        return new_audio

class ToTensor:
    def __call__(self, mel):
        mel = torch.Tensor(mel)
        return mel.view(1, *mel.size())
    
class ToTensorRaw:
    def __call__(self, audio):
        audio = torch.Tensor(audio)
        return audio

class MinMaxChunkScaler:
    def __init__(self, chunk=50):
        self.chunk = chunk
    
    def __call__(self, audio):
        max_, min_ = audio.max(), audio.min()
        len_ = len(audio) // self.chunk

        for i in range(self.chunk):
            X = audio[len_*i: len_*(i+1)]
            if (X.max() - X.min()) == 0: continue

            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (max_ - min_) + min_
            audio[len_*i: len_*(i+1)] = X_scaled
        return audio

class MinMaxChunkScaler:
    def __call__(self, audio):
        max_, min_ = audio.max(), audio.min()
        chunk = len(audio) // 10000
        len_ = len(audio) // chunk

        for i in range(chunk):
            X = audio[len_*i: len_*(i+1)]
            if (X.max() - X.min()) == 0: continue

            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (max_ - min_) + min_
            audio[len_*i: len_*(i+1)] = X_scaled
        return audio