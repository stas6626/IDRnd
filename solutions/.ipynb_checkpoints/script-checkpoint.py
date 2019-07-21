#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from IDRnD.utils import seed_everything
from IDRnD.augmentations import ToMellSpec, PadOrClip, ToTensor, Normalize_predef
from IDRnD.dataset import Test_Dataset
from IDRnD.resnet import resnet34
from IDRnD.pipeline import *


import pandas as pd
import numpy as np
import librosa
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from mag.experiment import Experiment
from IDRnD.dimka.networks.classifiers import TwoDimensionalCNNClassificationModel

seed_everything(0)


# In[2]:


dataset_dir = "."

eval_protocol_path = "protocol_test.txt"
eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
eval_protocol.columns = ["path", "key"]
eval_protocol["score"] = 0.0
# eval_protocol['path'] = eval_protocol['path'].apply(lambda x: os.path.join(dataset_dir, x))


# In[2]:


post_transform = transforms.Compose(
    [
        ToMellSpec(n_mels=128),
        librosa.power_to_db,
        PadOrClip(320),
        Normalize_predef(-29.6179, 16.6342),
        ToTensor(),
    ]
)

with Experiment({
    "data": {
        "_input_dim": 64,
        "_kfold_seed": 42,
        "_n_folds": 5,
        "_train_data_dir": "data/Training_Data/",
        "_train_df": "data/train_df.csv",
        "features": "mel_1024_512_64",
        "max_audio_length": 3,
        "p_aug": 0.3,
        "p_mixup": 0.0
    },
    "label": "2d_cnn",
    "network": {
        "aggregation_type": "max",
        "conv_base_depth": 32,
        "growth_rate": 1.3,
        "num_conv_blocks": 5,
        "output_dropout": 0.5,
        "start_deep_supervision_on": 2
    },
    "train": {
        "_save_every": 5,
        "accumulation_steps": 1,
        "batch_size": 50,
        "epochs": 7,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "scheduler": "1cycle_0.0001_0.005",
        "switch_off_augmentations_on": 6,
        "weight_decay": 0.0
    }
}) as experiment:
    device = "cuda"
    config = experiment.config
    model = TwoDimensionalCNNClassificationModel(experiment, device=device)


# ### predict

# In[3]:


# In[4]:


hm = Train()

test_dataset = Test_Dataset(np.array(eval_protocol["path"]), post_transform)

test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

model.eval()
preds = None
for i in range(5):
    state = torch.load(f'models/dimka_model_from_pretrain_fold_{i}.pt')
    model.load_state_dict(state)
    pred = hm.predict_on_test(test_loader, model).values
    if preds is None:
        preds = pred
    else:
        preds += pred

eval_protocol["score"] = pred/5
eval_protocol[["path", "score"]].to_csv("answers.csv", index=None)