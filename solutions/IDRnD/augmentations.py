import random

import numpy as np
import librosa
import torch


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
        if random.random() < self.p:
            # noise = numpy.random.normal(0, self.std, len(audio))
            return audio + np.random.normal(0, self.std, len(audio))
        else:
            return audio


class Shift:
    def __init__(self, shift, p=0.2):
        self.shift = int(shift)
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            return np.roll(audio, self.shift)
        else:
            return audio


class TimeStretch:
    def __init__(self, rate, p=0.2):
        self.rate = rate
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            return librosa.effects.time_stretch(audio, self.rate)
        else:
            return audio


class Distortion:
    def __init__(self, min_=-5000, max_=5000, p=0.2):
        self.min_ = min_
        self.max_ = max_
        self.p = p

    def __call__(self, audio):
        if random.random() < self.p:
            return np.clip(audio, self.min_, self.max_)
        else:
            return audio


class RandomParameter:
    def __init__(self, aug, parametres: list, p=0.2, num_params=1):
        self.aug = aug
        self.parametres = parametres
        self.p = p
        self.num_params = num_params

    def __call__(self, audio):
        if random.random() < self.p:

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
        self.sr = 16000

    def __call__(self, audio):
        if random.random() < self.p:
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
        mel = librosa.feature.melspectrogram(
            audio,
            sr=16000,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return mel


class PadOrClip:
    def __init__(self, pad_lenth, save_mean_and_var=True):
        self.pad_lenth = pad_lenth
        self.save_mean_and_var = save_mean_and_var

    def __call__(self, mel):
        if len(mel.shape) == 2:
            if mel.shape[1] < self.pad_lenth:
                if self.save_mean_and_var:
                    mean, std = np.mean(mel), np.std(mel)
                
                init_shape = mel.shape[1]
                mel = np.pad(
                    mel, [(0, 0), (0, self.pad_lenth - init_shape)], "constant"
                )
                if self.save_mean_and_var:
                    normal = np.random.normal(mean, std, size=(mel.shape[0], self.pad_lenth - init_shape))
                    mel[:, init_shape:] = normal
            else:
                right_limit = mel.shape[1] - self.pad_lenth
                shift = random.randint(0, right_limit)
                mel = mel[:, shift : shift + self.pad_lenth]
            return mel

        elif len(mel.shape) == 1:
            if mel.shape[0] < self.pad_lenth:
                mel = np.pad(mel, [0, self.pad_lenth - mel.shape[0]], "constant")

            else:
                right_limit = mel.shape[0] - self.pad_lenth
                shift = random.randint(0, right_limit)
                mel = mel[shift : shift + self.pad_lenth]
            return mel


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
            X = audio[len_ * i : len_ * (i + 1)]
            if (X.max() - X.min()) == 0:
                continue

            X_std = (X - X.min()) / (X.max() - X.min())
            X_scaled = X_std * (max_ - min_) + min_
            audio[len_ * i : len_ * (i + 1)] = X_scaled
        return audio
