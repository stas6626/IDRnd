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

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--experiment", type=str, required=True,
    help="path to an experiment"
)
parser.add_argument(
    "--test_df", required=True, type=str,
    help="path to test dataframe"
)
parser.add_argument(
    "--output_df", required=True, type=str,
    help="where to save resulting dataframe"
)
parser.add_argument(
    "--test_data_dir", required=True, type=str,
    help="path to test data directory"
)
parser.add_argument(
    "--batch_size", type=int, default=32,
    help="batch size used for prediction"
)
parser.add_argument(
    "--threshold", type=int,
    help="threshold used for prediction"
)
parser.add_argument(
    "--device", type=str, required=True,
    help="whether to train on cuda or cpu",
    choices=("cuda", "cpu")
)
parser.add_argument(
    "--num_workers", type=int, default=4,
    help="number of workers for data loader",
)

args = parser.parse_args()

test_df = pd.read_csv(args.test_df)

with Experiment(resume_from=args.experiment) as experiment:

    config = experiment.config

    audio_transform = AudioFeatures(config.data.features)

    all_predictions = []

    for fold in range(config.data._n_folds):

        print("\n\n   -----  Fold {}\n".format(fold))

        loader_kwargs = (
            {"num_workers": args.num_workers, "pin_memory": True}
            if torch.cuda.is_available() else {})

        test_loader = torch.utils.data.DataLoader(
            AntispoofDataset(
                audio_files=[
                    os.path.join(args.test_data_dir, fname)
                    for fname in test_df.fname.values],
                labels=None,
                transform=Compose([
                    LoadAudio(),
                    audio_transform,
                    DropFields(("audio", "filename", "sr")),
                ]),
            ),
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=make_collate_fn({"signal": audio_transform.padding_value}),
            **loader_kwargs
        )

        model = TwoDimensionalCNNClassificationModel(
                experiment, device=args.device)
        model.load_best_model(fold)
        model.eval()

        test_preds = model.predict(test_loader)
        all_predictions.append(test_preds)

test_predictions = np.mean(all_predictions, 0)

if args.threshold is not None:
    test_predictions = (test_predictions > args.threshold).astype(np.int32)

test_df["labels"] = test_predictions
test_df.to_csv(args.output_df, index=False)