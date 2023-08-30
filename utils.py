import librosa
from tqdm.auto import tqdm
import os
import numpy as np
from torch.utils.data import Dataset
import json

raw_splits = json.load(open("./MTT/magnatagatune.json", "r"))

splits = {"train": {}, "test": {}, "valid": {}}

for idx, row in tqdm(raw_splits.items(), total=len(raw_splits)):
    if row['split'] is not None and os.path.exists(f"./MTT/audios/{row['extra']['mp3_path']}"):
        splits[row['split']][idx] = row

labels = "guitar, classical, slow, techno, strings, drums, electronic, rock, fast, piano, ambient, beat, violin, " \
         "vocal, synth, female, indian, opera, male, singing, vocals, no vocals, harpsichord, loud, quiet, flute, " \
         "woman, male vocal, no vocal, pop, soft, sitar, solo, man, classic, choir, voice, new age, dance, " \
         "male voice, female vocal, beats, harp, cello, no voice, weird, country, metal, female voice, choral"\
    .split(", ")
label_count = len(labels)
label_to_idx = {l: i for i, l in enumerate(labels)}


class MTTDataset(Dataset):
    def __init__(self, split="train", sr=24000):
        self.X, self.Y = [], []
        for idx, row in tqdm(splits[split].items(), total=len(splits[split])):
            audio = librosa.load(f"./MTT/audios/{row['extra']['mp3_path']}", sr=sr)[0]
            if len(audio.shape) == 2:
                audio = audio.mean(1)
            self.X.append(audio)
            label = np.zeros(50)
            for la in row['y']:
                label[label_to_idx[la]] = 1
            self.Y.append(label)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
