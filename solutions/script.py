#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from IDRnD.utils import Train, seed_everything
from IDRnD.augmentations import ToMellSpec, PadOrClip, ToTensor, Normalize_predef
from IDRnD.dataset import Test_Dataset
<<<<<<< HEAD
from IDRnD.resnet import resnet34
=======
from IDRnD.resnet import resnet50
>>>>>>> 627a09b06e7379c4ea4e77aa8eca7902f2674035

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
<<<<<<< HEAD
eval_protocol.columns = ["path", "key"]
eval_protocol["score"] = 0.0
# eval_protocol['path'] = eval_protocol['path'].apply(lambda x: os.path.join(dataset_dir, x))
=======
eval_protocol.columns = ['path', 'key']
eval_protocol['score'] = 0.0
#eval_protocol['path'] = eval_protocol['path'].apply(lambda x: os.path.join(dataset_dir, x))
>>>>>>> 627a09b06e7379c4ea4e77aa8eca7902f2674035


# In[2]:


<<<<<<< HEAD
post_transform = transforms.Compose(
    [
        ToMellSpec(n_mels=128),
        librosa.power_to_db,
        PadOrClip(320),
        Normalize_predef(-29.6179, 16.6342),
        ToTensor(),
    ]
)
=======
post_transform = transforms.Compose([
    ToMellSpec(n_mels=128),
    Normalize_predef(-29.6179, 16.6342),
    librosa.power_to_db,
    PadOrClip(150),
    ToTensor(),
])
>>>>>>> 627a09b06e7379c4ea4e77aa8eca7902f2674035


# ### predict

# In[3]:


<<<<<<< HEAD
=======

>>>>>>> 627a09b06e7379c4ea4e77aa8eca7902f2674035
# In[4]:


hm = Train()

test_dataset = Test_Dataset(np.array(eval_protocol["path"]), post_transform)
<<<<<<< HEAD

test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

model = resnet34(num_classes=1).cuda()

model.eval()
model.load_state_dict(torch.load("models/resnet_34.pt9"))
=======
#test_dataset = Test_Dataset(X[:300], post_transform)

test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

model = resnet50(num_classes=1).cuda()

#model.load_state_dict(torch.load('models/simple_old_conv.pt'))
#model_dst = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
#torch.save(model_dst.module.state_dict(),  'models/kaggle2_nonparallel.pt') 
model.eval()
model.load_state_dict(torch.load('models/resnet_34_5ep.pt'))
>>>>>>> 627a09b06e7379c4ea4e77aa8eca7902f2674035
pred = hm.predict_on_test(test_loader, model)


# In[7]:


eval_protocol["score"] = pred.values
<<<<<<< HEAD
eval_protocol[["path", "score"]].to_csv("answers.csv", index=None)
=======
eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
>>>>>>> 627a09b06e7379c4ea4e77aa8eca7902f2674035
