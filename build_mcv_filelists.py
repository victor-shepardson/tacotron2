import sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm

import layers
from hparams import create_hparams
from utils import load_audio_to_torch

process_audio = sys.argv[1] if len(sys.argv)>1 else False

hparams = create_hparams()
stft = layers.TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length,
    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
    hparams.mel_fmax)

data_root = '../data/mozilla_common_voice/eo'

data = pd.read_csv(os.path.join(data_root, 'validated.tsv'), sep='\t')
# convert client_id to speaker id and discard infrequent speakers
speaker_map = {
    id:(i if count>=30 else -1)
    for i, (id, count) in enumerate(data.client_id.value_counts().iteritems())}
data['speaker'] = data.client_id.map(speaker_map)
data = data[data.speaker>=0]
print('speaker distribution:')
print(data.speaker.value_counts())
# respect orginal validation split so models trained with it can be used for warm start
valid_data = pd.read_csv(os.path.join(data_root, 'dev.tsv'), sep='\t')
valid_paths = (
    valid_data[valid_data.client_id.isin(data.client_id)].path
    .sample(200, replace=True, random_state=0))

# train/valid split
is_val = data.path.isin(valid_paths)
train_data, val_data = data[~is_val], data[is_val]

# audio features
def gen_spectra(paths):
    for fname in tqdm(paths, desc='processing audio'):
        s = stft.mel_spectrogram(
            load_audio_to_torch(f'{data_root}/clips/{fname}.mp3', hparams.sampling_rate, wav_scale=False)[0]
            .unsqueeze(0)
        ).squeeze(0).numpy()
        # trim leading/trailing silence
        v = (np.max(s, axis=0)>-5).astype(np.float)
        s = s[:, np.cumsum(v) * np.cumsum(v[::-1])[::-1] > 0]
        yield s

# save spectra with np.save
if process_audio:
    for fname, s in zip(data.path, gen_spectra(data.path)):
        np.save(f'{data_root}/spect/{fname}', s)

# write filelists
for data, dest in (
        (train_data, 'filelists/mcv_eo_train_filelist.txt'),
        (val_data, 'filelists/mcv_eo_val_filelist.txt')):
    with open(dest, 'w') as fl:
        for fname, text, speaker, lang in zip(
                tqdm(data.path, desc='writing filelist'),
                data.sentence, data.speaker, np.zeros(len(data), dtype=int)):
            fl.write(f'{data_root}/spect/{fname}.npy|{text}|{speaker}|{lang}\n')

# data_root = '../data/mozilla_common_voice/eo'
# datasets = {
#     'train.tsv': 'filelists/mcv_eo_train_filelist.txt',
#     'test.tsv': 'filelists/mcv_eo_test_filelist.txt',
#     'dev.tsv': 'filelists/mcv_eo_val_filelist.txt'
# }
#
# for src, dest in datasets.items():
#     src = os.path.join(data_root, src)
#     data = pd.read_csv(src, sep='\t')
#
#     with open(dest, 'w') as fl:
#         for fname, text in zip(data.path, data.sentence):
#             fl.write(f'{data_root}/clips/{fname}.mp3|{text}\n')
