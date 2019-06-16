import os
import argparse
import glob

import pandas as pd

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--data_directory", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)

args = parser.parse_args()


human_directory = os.path.join(args.data_directory, "human")
spoof_directory = os.path.join(args.data_directory, "spoof")

human_files = sorted(glob.iglob(
    os.path.join(human_directory, "**/*.wav"), recursive=True))
spoof_files = sorted(glob.iglob(
    os.path.join(spoof_directory, "**/*.wav"), recursive=True))

human_files = [os.path.join("human", os.path.basename(p)) for p in human_files]
spoof_files = [os.path.join("spoof", os.path.basename(p)) for p in spoof_files]

train_df = pd.DataFrame({
    "fname": human_files + spoof_files,
    "labels": [1] * len(human_files) + [0] * len(spoof_files)
})

train_df.to_csv(args.output_file, index=False)