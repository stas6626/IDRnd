import random
import os
from .augmentations import PadOrClip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import librosa
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path



class FATTrainDataset(Dataset):
    def __init__(self, mels, labels, transforms):
        super().__init__()
        self.mels = mels
        self.labels = labels
        self.transforms = transforms
        
    def __len__(self):
        return len(self.mels)
    
    def __getitem__(self, idx):
        # crop 1sec
        image = Image.fromarray(self.mels[idx], mode='RGB')        
        time_dim, base_dim = image.size
        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        image = self.transforms(image).div_(255)
        
        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        
        return image, label
    

class FATTestDataset(Dataset):
    def __init__(self, fnames, mels, transforms, tta=5):
        super().__init__()
        self.fnames = fnames
        self.mels = mels
        self.transforms = transforms
        self.tta = tta
        
    def __len__(self):
        return len(self.fnames) * self.tta
    
    def __getitem__(self, idx):
        new_idx = idx % len(self.fnames)
        
        image = Image.fromarray(self.mels[new_idx], mode='RGB')
        time_dim, base_dim = image.size
        crop = random.randint(0, time_dim - base_dim)
        image = image.crop([crop, 0, crop + base_dim, base_dim])
        image = self.transforms(image).div_(255)

        fname = self.fnames[new_idx]
        
        return image, fname
    
class Dataset_Train(Dataset):
    def __init__(self, train_csv, labels, transforms=None, preload=False):
        super().__init__()
        self.train_csv = train_csv
        self.labels = labels
        self.transforms = transforms
        self.preload = preload
        self.sr = 16000
        
    def __len__(self):
        return len(self.train_csv)
    
    def __getitem__(self, idx):
        try:            
            if self.preload:
                audio = self.train_csv[idx] ##in this case its will be just a list on numpy arrays
            else:
                audio, _ = librosa.core.load(self.train_csv.iloc[idx, 0], sr=self.sr)

            if self.transforms:
                audio = self.transforms(audio)

        except Exception as exp:
            logging.debug(f"actuall idx is:{idx} and error is {exp}")
            return self[idx+1]
            
        label = self.labels[idx]
        label = torch.Tensor([label]).float()
        
        return audio, label
    

class Dataset_Test(Dataset):
    def __init__(self, train_csv, names, transforms=None, preload=False, tta=5):
        super().__init__()
        self.train_csv = train_csv
        self.names = names
        self.transforms = transforms
        self.preload = preload
        self.sr = 44100
        self.tta = tta
        
    def __len__(self):
        return len(self.train_csv) * self.tta
    
    def __getitem__(self, idx):
        idx = idx//self.tta
        
        if self.preload:
            audio = self.train_csv[idx] ##in this case its will be just a list on numpy arrays
        else:
            audio, _ = librosa.core.load(self.train_csv.iloc[idx, 0], sr=self.sr)

        if self.transforms:
            audio = self.transforms(audio)

        name = self.names[idx]

        return audio, name


class Dataset_Train_Preload(Dataset):
    def __init__(self, labels, folder, transforms=None):
        super().__init__()
        self.labels = labels
        self.folder = folder
        self.transforms = transforms
        self.pathes =  [dict() for _ in range(len(self))]
        self.load()
    
    def load(self):
        data = os.listdir(self.folder)
        for name in data:
            if name == ".ipynb_checkpoints": continue
            idx = int(name.split("_")[1].split(".")[0])
            if name not in self.pathes[idx]:
                self.pathes[idx][name] = 0
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if idx % 1000 == 0:
            self.load()
        min_ = min(self.pathes[idx].keys(), key=lambda x: self.pathes[idx][x])
        path = os.path.join(self.folder, min_)
        self.pathes[idx][min_] += 1
        
        mel = np.load(path, allow_pickle=True)
        if self.transforms is not None:
            mel = self.transforms(mel)
        
        label = self.labels[idx]
        label = torch.from_numpy(label).float()
        
        return mel, label
    
class Dataset_Train_Mixmatch(Dataset):
    def __init__(self, base_dataset, mix_samples=2, mix_rounds=1, pad_lenth=100000, post_transform=None):
        super().__init__()
        self.base_dataset = base_dataset
        self.mix_samples = mix_samples
        self.samples_to_process = set(range(len(self.base_dataset)))
        self.pad = PadOrClip(pad_lenth)
        self.post_transform = post_transform
        self.mix_rounds = mix_rounds

    def __len__(self):
        return len(self.base_dataset) + (len(self.base_dataset)//self.mix_samples)*self.mix_rounds

    def __getitem__(self, idx):
        if idx < len(self.base_dataset):
            audio, target = self.base_dataset[idx]
            if self.post_transform is not None:
                audio = self.post_transform(audio)
            return audio, target

        else:
            if len(self.samples_to_process) < len(self.base_dataset)//self.mix_samples:
                self.samples_to_process = set(range(len(self.base_dataset)))

            samples = random.sample(self.samples_to_process, self.mix_samples)
            self.samples_to_process.difference_update(set(samples))
            audio = None
            target = torch.zeros(80)

            for sample in samples:
                audio_, target_ = self.base_dataset[sample]
                audio_ = self.pad(audio_)
                if audio is None:
                    audio = audio_ / len(samples)
                else:
                    audio += audio_ / len(samples)
                target += target_

            if self.post_transform is not None:
                audio = self.post_transform(audio)
            
            target = torch.clamp(target, 0, 1)

            return audio, target



def get_tables():
    dataset_dir = Path('../data')
    preprocessed_dir = Path('../data/preproc')

    csvs = {
        'train_curated': dataset_dir / 'train_curated.csv',
        'train_noisy': dataset_dir / 'train_noisy.csv',
        #'train_noisy': preprocessed_dir / 'trn_noisy_best50s.csv',
        'sample_submission': dataset_dir / 'sample_submission.csv',
    }

    dataset = {
        'train_curated': dataset_dir / 'train_curated',
        'train_noisy': dataset_dir / 'train_noisy',
        'test': dataset_dir / 'test',
    }

    train_curated = pd.read_csv(csvs['train_curated'])
    train_noisy = pd.read_csv(csvs['train_noisy'])

    train_curated["fname"] = train_curated["fname"].apply(lambda x: f"{dataset_dir}/{x}")
    train_noisy["fname"] = train_noisy["fname"].apply(lambda x: f"{dataset_dir}/{x}")

    train_df = pd.concat([train_curated, train_noisy], sort=True, ignore_index=True)
    test_df = pd.read_csv(csvs['sample_submission'])
    names = test_df["fname"].apply(lambda x: f"{dataset_dir}/{x}")

    labels = test_df.columns[1:].tolist()
    num_classes = 80

    y_curated_np = np.zeros((len(train_curated), num_classes)).astype(int)
    for i, row in enumerate(train_curated['labels'].str.split(',')):
        for label in row:
            idx = labels.index(label)
            y_curated_np[i, idx] = 1

    y_noisy_np = np.zeros((len(train_noisy), num_classes)).astype(int)
    for i, row in enumerate(train_noisy['labels'].str.split(',')):
        for label in row:
            idx = labels.index(label)
            y_noisy_np[i, idx] = 1
    return names, y_curated_np, y_noisy_np, train_curated, train_noisy, test_df, labels
