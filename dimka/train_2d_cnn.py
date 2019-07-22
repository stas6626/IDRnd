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
import mag

from datasets.antispoof_dataset import AntispoofDataset
from networks.classifiers import TwoDimensionalCNNClassificationModel
from ops.folds import train_validation_data_stratified
from ops.transforms import (
    Compose, DropFields, LoadAudio,
    AudioFeatures, MapLabels, RenameFields,
    MixUp, SampleSegment, SampleLongAudio,
    AudioAugmentation, ShuffleAudio, CutOut, Identity)
from ops.utils import load_json, compute_inverse_eer
from ops.padding import make_collate_fn

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

mag.use_custom_separator("-")

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    "--train_df", required=True, type=str,
    help="path to train dataframe"
)
parser.add_argument(
    "--train_data_dir", required=True, type=str,
    help="path to train data"
)
parser.add_argument(
    "--resume", action="store_true", default=False,
    help="allow resuming even if experiment exists"
)
parser.add_argument(
    "--log_interval", default=10, type=int,
    help="how frequently to log batch metrics"
    "in terms of processed batches"
)
parser.add_argument(
    "--batch_size", type=int, default=64,
    help="minibatch size"
)
parser.add_argument(
    "--max_audio_length", type=int, default=10,
    help="max audio length in seconds. For longer clips are sampled"
)
parser.add_argument(
    "--lr", default=0.01, type=float,
    help="starting learning rate"
)
parser.add_argument(
    "--max_samples", type=int,
    help="maximum number of samples to use"
)
parser.add_argument(
    "--epochs", default=100, type=int,
    help="number of epochs to train"
)
parser.add_argument(
    "--scheduler", type=str, default="steplr_1_0.5",
    help="scheduler type",
)
parser.add_argument(
    "--accumulation_steps", type=int, default=1,
    help="number of gradient accumulation steps",
)
parser.add_argument(
    "--save_every", type=int, default=1,
    help="how frequently to save a model",
)
parser.add_argument(
    "--device", type=str, required=True,
    help="whether to train on cuda or cpu",
    choices=("cuda", "cpu")
)
parser.add_argument(
    "--aggregation_type", type=str, required=True,
    help="how to aggregate outputs",
    choices=("max", "rnn")
)
parser.add_argument(
    "--num_conv_blocks", type=int, default=5,
    help="number of conv blocks"
)
parser.add_argument(
    "--start_deep_supervision_on", type=int, default=2,
    help="from which layer to start aggregating features for classification"
)
parser.add_argument(
    "--conv_base_depth", type=int, default=64,
    help="base depth for conv layers"
)
parser.add_argument(
    "--growth_rate", type=float, default=2,
    help="how quickly to increase the number of units as a function of layer"
)
parser.add_argument(
    "--weight_decay", type=float, default=1e-5,
    help="weight decay"
)
parser.add_argument(
    "--output_dropout", type=float, default=0.0,
    help="output dropout"
)
parser.add_argument(
    "--p_mixup", type=float, default=0.0,
    help="probability of the mixup augmentation"
)
parser.add_argument(
    "--p_aug", type=float, default=0.0,
    help="probability of audio augmentation"
)
parser.add_argument(
    "--switch_off_augmentations_on", type=int, default=20,
    help="on which epoch to remove augmentations"
)
parser.add_argument(
    "--features", type=str, required=True,
    help="feature descriptor"
)
parser.add_argument(
    "--optimizer", type=str, required=True,
    help="which optimizer to use",
    choices=("adam", "momentum")
)
parser.add_argument(
    "--folds", type=int, required=True, nargs="+",
    help="which folds to use"
)
parser.add_argument(
    "--n_folds", type=int, default=4,
    help="number of folds"
)
parser.add_argument(
    "--kfold_seed", type=int, default=42,
    help="kfold seed"
)
parser.add_argument(
    "--num_workers", type=int, default=4,
    help="number of workers for data loader",
)
parser.add_argument(
    "--label", type=str, default="2d_cnn",
    help="optional label",
)
args = parser.parse_args()

audio_transform = AudioFeatures(args.features)

with Experiment({
    "network": {
        "num_conv_blocks": args.num_conv_blocks,
        "start_deep_supervision_on": args.start_deep_supervision_on,
        "conv_base_depth": args.conv_base_depth,
        "growth_rate": args.growth_rate,
        "output_dropout": args.output_dropout,
        "aggregation_type": args.aggregation_type
    },
    "data": {
        "features": args.features,
        "_n_folds": args.n_folds,
        "_kfold_seed": args.kfold_seed,
        "_input_dim": audio_transform.n_features,
        "p_mixup": args.p_mixup,
        "p_aug": args.p_aug,
        "max_audio_length": args.max_audio_length,
        "_train_df": args.train_df,
        "_train_data_dir": args.train_data_dir
    },
    "train": {
        "accumulation_steps": args.accumulation_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "scheduler": args.scheduler,
        "optimizer": args.optimizer,
        "epochs": args.epochs,
        "_save_every": args.save_every,
        "weight_decay": args.weight_decay,
        "switch_off_augmentations_on": args.switch_off_augmentations_on
    },
    "label": args.label
}, implicit_resuming=args.resume) as experiment:

    config = experiment.config
    print()
    print("     ////// CONFIG //////")
    print(experiment.config)

    train_df = pd.read_csv(args.train_df)

    if args.max_samples:
        train_df = train_df.sample(args.max_samples).reset_index(drop=True)

    if "cluster" in train_df:
        groups = train_df.cluster.tolist()
    else:
        groups = None

    splits = list(train_validation_data_stratified(
        train_df.fname, train_df.labels,
        config.data._n_folds, config.data._kfold_seed, groups))

    for fold in args.folds:

        print("\n\n   -----  Fold {}\n".format(fold))

        train, valid = splits[fold]

        loader_kwargs = (
            {"num_workers": args.num_workers, "pin_memory": True}
            if torch.cuda.is_available() else {})

        experiment.register_directory("checkpoints")
        experiment.register_directory("predictions")

        train_loader = torch.utils.data.DataLoader(
            AntispoofDataset(
                audio_files=[
                    os.path.join(args.train_data_dir, fname)
                    for fname in train_df.fname.values[train]],
                labels=train_df.labels.values[train],
                transform=Compose([
                    LoadAudio(),
                    SampleLongAudio(max_length=args.max_audio_length),
                    MixUp(p=args.p_mixup),
                    AudioAugmentation(p=args.p_aug),
                    audio_transform,
                    DropFields(("audio", "sr")),
                ]),
                clean_transform=Compose([
                    LoadAudio(),
                    SampleLongAudio(max_length=args.max_audio_length)
                ])
            ),
            shuffle=True,
            drop_last=True,
            batch_size=config.train.batch_size,
            collate_fn=make_collate_fn({"signal": audio_transform.padding_value}),
            **loader_kwargs
        )

        valid_loader = torch.utils.data.DataLoader(
            AntispoofDataset(
                audio_files=[
                    os.path.join(args.train_data_dir, fname)
                    for fname in train_df.fname.values[valid]],
                labels=train_df.labels.values[valid],
                transform=Compose([
                    LoadAudio(),
                    audio_transform,
                    DropFields(("audio", "sr")),
                ])
            ),
            shuffle=False,
            batch_size=config.train.batch_size,
            collate_fn=make_collate_fn({"signal": audio_transform.padding_value}),
            **loader_kwargs
        )

        model = TwoDimensionalCNNClassificationModel(
            experiment, device=args.device)

        scores = model.fit_validate(
            train_loader, valid_loader,
            epochs=experiment.config.train.epochs, fold=fold,
            log_interval=args.log_interval
        )

        best_metric = max(scores)
        experiment.register_result("fold{}.metric".format(fold), best_metric)

        torch.save(
            model.state_dict(),
            os.path.join(
                experiment.checkpoints,
                "fold_{}".format(fold),
                "final_model.pth")
        )

        model.load_best_model(fold)

        # validation

        val_preds = model.predict(valid_loader)
        val_predictions_df = pd.DataFrame(
            val_preds, columns=["labels"])
        val_predictions_df["fname"] = train_df.fname[valid].values
        val_predictions_df.to_csv(
            os.path.join(
                experiment.predictions,
                "val_preds_fold_{}.csv".format(fold)
            ),
            index=False
        )
        del val_predictions_df

        if args.device == "cuda":
            torch.cuda.empty_cache()

    # global metric

    if all(
        "fold{}".format(k) in experiment.results.to_dict()
        for k in range(config.data._n_folds)):

        val_df_files = [
            os.path.join(
                experiment.predictions,
                "val_preds_fold_{}.csv".format(fold)
            )
            for fold in range(config.data._n_folds)
        ]

        val_predictions_df = pd.concat([
            pd.read_csv(file) for file in val_df_files]).reset_index(drop=True)

        labels = np.asarray([
            item["labels"][0] for item in AntispoofDataset(
                audio_files=train_df.fname.tolist(),
                labels=train_df.labels.values,
                transform=None
            )
        ])

        val_labels_df = pd.DataFrame(
            labels, columns=["labels"])
        val_labels_df["fname"] = train_df.fname

        assert set(val_predictions_df.fname) == set(val_labels_df.fname)

        val_predictions_df.sort_values(by="fname", inplace=True)
        val_labels_df.sort_values(by="fname", inplace=True)

        metric = compute_inverse_eer(
            val_labels_df.drop("fname", axis=1).values,
            val_predictions_df.drop("fname", axis=1).values
        )

        experiment.register_result("metric", metric)