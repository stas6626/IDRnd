import json

import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
import librosa


def compute_inverse_eer(y_true, scores):

    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    try:
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    except ValueError:
        eer = 0.0  # TODO: fix this
    return 1 - eer


def load_json(file):
    with open(file, "r") as f:
        return json.load(f)


def make_mel_filterbanks(descriptor, sr=44100):

    name, *args = descriptor.split("_")

    n_fft, hop_size, n_mel = args
    n_fft = int(n_fft)
    hop_size = int(hop_size)
    n_mel = int(n_mel)

    filterbank = librosa.filters.mel(
        sr=sr, n_fft=n_fft, n_mels=n_mel,
        fmin=5, fmax=None
    ).astype(np.float32)

    return filterbank


def is_mel(descriptor):
    return descriptor.startswith("mel")


def is_stft(descriptor):
    return descriptor.startswith("stft")


def compute_torch_stft(audio, descriptor):

    name, *args = descriptor.split("_")

    n_fft, hop_size, *rest = args
    n_fft = int(n_fft)
    hop_size = int(hop_size)

    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_size,
        window=torch.hann_window(n_fft, device=audio.device)
    )

    stft = torch.sqrt((stft ** 2).sum(-1))

    return stft

