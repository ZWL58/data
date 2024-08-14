import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
import torch
import pre_aud as pa
import pandas as pd
import numpy as np

class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Absolute file path of the audio file - concatenate the audio directory with
        # the relative path
        audio_file = self.df.loc[idx, 'relative_path']
        # Get the Class ID
        class_id = self.df.loc[idx, 'label']

        aud = pa.AudioUtil.open(audio_file)
        reaud = pa.AudioUtil.resample(aud, self.sr)
        rechan = pa.AudioUtil.rechannel(reaud, self.channel)

        dur_aud = pa.AudioUtil.pad_trunc(rechan, self.duration)

        sgram = pa.AudioUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)

        return sgram, class_id



def Dataloader():
    # Path to read the csv file
    metadata_file = 'Path'
    df = pd.read_csv(metadata_file)
    df['relative_path'] = '' + '/' + df['class'] + '/' + df['audio_name'].astype(str)
    data_path = df[['relative_path', 'label', 'class']]
    num_class = df['class']
    num_class = set(num_class)

    myds = SoundDS(df, data_path)
    myds[0]

    # Random split of 80:20 between training and validation
    num_items = len(myds)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(myds, [num_train, num_val])

    # Create training and validation data loaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    return train_dl,val_dl,num_class





