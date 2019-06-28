import sys
import subprocess

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


# install("mag")


import os
import gc
import argparse
import json
import math
from functools import partial

import pandas as pd
import numpy as np
import torch
from mag.experiment import Experiment
from mag.utils import green, bold
import mag

from datasets.antispoof_dataset import AntispoofDataset
from networks.classifiers import TwoDimensionalCNNClassificationModel
from ops.transforms import (
    Compose, DropFields, LoadAudio,
    AudioFeatures, MapLabels, RenameFields,
    MixUp, SampleSegment, SampleLongAudio,
    AudioAugmentation, FlipAudio, ShuffleAudio)
from ops.padding import make_collate_fn


mag.use_custom_separator("-")

eval_protocol_path = "protocol_test.txt"
eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
eval_protocol.columns = ['path', 'key']
eval_protocol['score'] = 0.0


num_workers = 0
batch_size = 10
device = "cuda"

experiment = "experiments/mel_1024_512_64-10-0.0-0.0-2d_cnn-max-64-1.3-5-0.0-2-1-50-15-0.001-adam-1cycle_0.0001_0.005-50-0.0"

folds = [0, 1]

predictions = []

with Experiment(resume_from=experiment) as experiment:

    config = experiment.config

    audio_transform = AudioFeatures(config.data.features)

    for fold in folds:

        print("\n\n   -----  Fold {}\n".format(fold))

        test_loader = torch.utils.data.DataLoader(
            AntispoofDataset(
                audio_files=np.array(eval_protocol["path"]),
                labels=None,
                transform=Compose([
                    LoadAudio(),
                    audio_transform,
                    DropFields(("audio", "sr")),
                ]),
            ),
            shuffle=False,
            batch_size=batch_size,
            collate_fn=make_collate_fn({"signal": audio_transform.padding_value}),
            num_workers=num_workers
        )

        model = TwoDimensionalCNNClassificationModel(experiment, device=device)
        model.load_best_model(fold)

        fold_predictions = model.predict(test_loader)
        predictions.append(fold_predictions)

predictions = np.mean(predictions, axis=0)


eval_protocol["score"] = predictions
eval_protocol[["path", "score"]].to_csv("answers.csv", index=None)
