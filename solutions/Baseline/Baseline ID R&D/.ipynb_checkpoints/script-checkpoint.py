import os
import sys
import glob
import tqdm
import keras
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

# Local file import
import DftSpectrogram

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.logging.set_verbosity(tf.logging.ERROR)
import librosa

print("Done!")

dataset_dir = "."

def get_feature(wav_path, length=66000, random_start=False):
    try:
        x, sr = librosa.load(wav_path, sr=None)
        assert sr == 16000
        if length > len(x):
            x = np.concatenate([x] * int(np.ceil(length/len(x))))
        if random_start:
            x = x[random.randint(0, len(x) - length):]
        feature = x[:length]
        return feature / np.max(np.abs(feature))
    except Exception as e:
        print("Error with getting feature from %s: %s" % (wav_path, str(e)))
        return None

class DevDataGenerator(keras.utils.Sequence):
    def __init__(self, human_paths, spoof_paths, feature_extractor):
        self.human_paths = human_paths
        self.spoof_paths = spoof_paths
        self.paths = human_paths + spoof_paths
        self.labels = [0] * len(human_paths) + [1] * len(spoof_paths)
        self.feature_extractor = feature_extractor
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        feature = self.feature_extractor(self.paths[index])[np.newaxis, ..., np.newaxis]
        return feature, keras.utils.to_categorical([self.labels[index]], num_classes=2)

dev_get_feature = lambda path: get_feature(path, length=66000, random_start=False)

wav_paths = sorted(glob.glob('./Testing_Data/*.wav', recursive=True))
feature = get_feature(random.choice(wav_paths), length=66000, random_start=True)

dft_conf = {"length": 512,
            "shift": 256,
            "nfft": 512,
            "mode": 'log',
            "normalize_feature": True}

inputs = keras.layers.Input(shape=(None, 1))
outputs = DftSpectrogram.DftSpectrogram(**dft_conf)(inputs)
model = keras.models.Model(inputs=inputs, outputs=outputs)
result = model.predict(feature[np.newaxis, ..., np.newaxis])
mobile_net_v2 = keras.applications.MobileNetV2(input_shape=result[0].shape, weights=None, classes=2)

inputs = keras.layers.Input(shape=(None, 1))
outputs = DftSpectrogram.DftSpectrogram(**dft_conf)(inputs)

model = keras.models.Model(inputs=inputs, outputs=mobile_net_v2(outputs))


model.load_weights("./baseline_model")

eval_protocol_path = "protocol_test.txt"
eval_protocol = pd.read_csv(eval_protocol_path, sep=" ", header=None)
eval_protocol.columns = ['path', 'key']
eval_protocol['score'] = 0.0

print(eval_protocol.shape)
print(eval_protocol.sample(5).head())

for protocol_id, protocol_row in tqdm.tqdm(list(eval_protocol.iterrows())):
    feature = dev_get_feature(os.path.join(dataset_dir, protocol_row['path']))[np.newaxis, ..., np.newaxis]
    score = model.predict(feature)
    eval_protocol.at[protocol_id, 'score'] = score[0][0]
eval_protocol[['path', 'score']].to_csv('answers.csv', index=None)
print(eval_protocol.sample(5).head())
