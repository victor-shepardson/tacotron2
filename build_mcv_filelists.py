import sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm

import layers
from hparams import create_hparams
from utils import load_audio_to_torch

process_audio = sys.argv[1] if len(sys.argv)>1 else False

data_root = '../data/mozilla_common_voice'
langs = [d for d in os.listdir(data_root) if not d.startswith('.')]
print(f'found {len(langs)} languages: {langs}')
min_speaker_samples = 100
val_size = 200*len(langs)

hparams = create_hparams()
stft = layers.TacotronSTFT(
    hparams.filter_length, hparams.hop_length, hparams.win_length,
    hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
    hparams.mel_fmax)

def gen_tables(fname):
    for i,l in enumerate(langs):
        data = pd.read_csv(os.path.join(data_root, l, fname), sep='\t')
        data['lang'] = l
        data['lang_idx'] = i
        yield data

data = pd.concat(gen_tables('validated.tsv')).reset_index(drop=True)
# convert client_id to speaker id and discard infrequent speakers
speaker_map = {
    id:(i if count>=min_speaker_samples else -1)
    for i, (id, count) in enumerate(data.client_id.value_counts().iteritems())}
data['speaker'] = data.client_id.map(speaker_map)
data = data[data.speaker>=0]
# print('speaker distribution:')
# print(data.speaker.value_counts())
print(f'found {data.speaker.nunique()} speakers')

# original validation split tests generalization across speakers (for recognition)
# for multi-speaker TTS model, we want to test across sentences within speakers
is_val = pd.Series(index=data.index, dtype=np.bool)
# stratified by language
for _,g in data.groupby('lang'):
    val_idxs = g.sample(val_size//len(langs), replace=False, random_state=0).index
    is_val[val_idxs] = True
train_data, val_data = data[~is_val], data[is_val]


# audio features
def gen_spectra(data):
    for fname, lang in zip(tqdm(data.path, desc='processing audio'), data.lang):
        s = stft.mel_spectrogram(
            load_audio_to_torch(f'{data_root}/{lang}/clips/{fname}.mp3', hparams.sampling_rate, wav_scale=False)[0]
            .unsqueeze(0)
        ).squeeze(0).numpy()
        # trim leading/trailing silence
        spectral_peaks = np.max(s[3:], axis=0)
        loud = np.argwhere((spectral_peaks > np.max(spectral_peaks)-3)).squeeze()
        lo, hi = max(0, loud[0]-16), min(s.shape[1], loud[-1]+32)
        yield s[:, lo:hi]

# save spectra with np.save
if process_audio:
    for lang in langs:
        path = os.path.join(data_root, lang, 'spect')
        if not os.path.exists(path):
            os.mkdir(path)

    for fname, lang, s in zip(data.path, data.lang, gen_spectra(data)):
        np.save(f'{data_root}/{lang}/spect/{fname}', s)

# write filelists
for data, dest in (
        (train_data, 'filelists/mcv_train_filelist.txt'),
        (val_data, 'filelists/mcv_val_filelist.txt')):
    with open(dest, 'w') as fl:
        for fname, text, speaker, lang, lang_idx in zip(
                tqdm(data.path, desc='writing filelist'),
                data.sentence, data.speaker, data.lang, data.lang_idx):
            fl.write(f'{data_root}/{lang}/spect/{fname}.npy|{text}|{speaker}|{lang_idx}\n')


# import sys, os
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
#
# import layers

# from hparams import create_hparams
# from utils import load_audio_to_torch
#
# process_audio = sys.argv[1] if len(sys.argv)>1 else False
#
# hparams = create_hparams()
# stft = layers.TacotronSTFT(
#     hparams.filter_length, hparams.hop_length, hparams.win_length,
#     hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
#     hparams.mel_fmax)
#
# data_root = '../data/mozilla_common_voice/eo'
#
# data = pd.read_csv(os.path.join(data_root, 'validated.tsv'), sep='\t')
# # convert client_id to speaker id and discard infrequent speakers
# speaker_map = {
#     id:(i if count>=30 else -1)
#     for i, (id, count) in enumerate(data.client_id.value_counts().iteritems())}
# data['speaker'] = data.client_id.map(speaker_map)
# data = data[data.speaker>=0]
# print('speaker distribution:')
# print(data.speaker.value_counts())
# # respect orginal validation split so models trained with it can be used for warm start
# valid_data = pd.read_csv(os.path.join(data_root, 'dev.tsv'), sep='\t')
# valid_paths = (
#     valid_data[valid_data.client_id.isin(data.client_id)].path
#     .sample(200, replace=True, random_state=0))
#
# # train/valid split
# is_val = data.path.isin(valid_paths)
# train_data, val_data = data[~is_val], data[is_val]
#
# # audio features
# def gen_spectra(paths):
#     for fname in tqdm(paths, desc='processing audio'):
#         s = stft.mel_spectrogram(
#             load_audio_to_torch(f'{data_root}/clips/{fname}.mp3', hparams.sampling_rate, wav_scale=False)[0]
#             .unsqueeze(0)
#         ).squeeze(0).numpy()
#         # trim leading/trailing silence
#         v = (np.max(s, axis=0)>-5).astype(np.float)
#         s = s[:, np.cumsum(v) * np.cumsum(v[::-1])[::-1] > 0]
#         yield s
#
# # save spectra with np.save
# if process_audio:
#     for fname, s in zip(data.path, gen_spectra(data.path)):
#         np.save(f'{data_root}/spect/{fname}', s)
#
# # write filelists
# for data, dest in (
#         (train_data, 'filelists/mcv_eo_train_filelist.txt'),
#         (val_data, 'filelists/mcv_eo_val_filelist.txt')):
#     with open(dest, 'w') as fl:
#         for fname, text, speaker, lang in zip(
#                 tqdm(data.path, desc='writing filelist'),
#                 data.sentence, data.speaker, np.zeros(len(data), dtype=int)):
#             fl.write(f'{data_root}/spect/{fname}.npy|{text}|{speaker}|{lang}\n')
#
