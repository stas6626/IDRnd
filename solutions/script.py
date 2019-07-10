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


# ### predict

# In[3]:


# In[4]:


hm = Train()

test_dataset = Test_Dataset(np.array(eval_protocol["path"]), post_transform)

test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

model = resnet34(num_classes=1).cuda()

model.eval()
model.load_state_dict(torch.load("models/resnet_34_common_voice.pt"))
pred = hm.predict_on_test(test_loader, model).values


eval_protocol["score"] = pred
eval_protocol[["path", "score"]].to_csv("answers.csv", index=None)