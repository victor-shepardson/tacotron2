import sys, os, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from fire import Fire

import layers
from hparams import create_hparams
from audio_processing import get_spectrum
from text.cleaners import multi_cleaners

"""Preprocess audio and build filelists for both tacotron2 and waveglow.
Assumes waveglow is nested in tacotron2 directory and not the other way around."""

# debug = bool(sys.argv[2]) if len(sys.argv)>2 else False

def path_to_id(p):
    return int(p.stem.split('_')[0][1:])

def main(
        process_audio=False, # precompute spectra
        write_wav=False, # also store the trimmed raw audio as .npy
        single_speaker=None, # use the single given ID
        remove_noise=False, # spectral noise removal
        data_root='../data/vctk/VCTK-Corpus',
        prefix='vctk', # for output filenames
        whitelist_file=None,
        val_size=100,
        hparams='',
        device='cpu',
        # threads=1 # soundfile seems not-threadsafe
    ):
    data_root = Path(data_root)#.expanduser().resolve()
    wav_files = {p.stem:p
        for p in tqdm(data_root.joinpath('wav48').rglob('*.wav'), desc='wav')}
    txt_files = {p.stem:p
        for p in tqdm(data_root.joinpath('txt').rglob('*.txt'), desc='text')}

    data = pd.read_csv(
        data_root.joinpath('speaker-info.txt'), sep='[ |\t]+', engine='python')
    # use only examples with corresponding wav+txt and with metadata for that speaker
    # files = [
    #     (wav_files[k], txt_files[k]) for k in wav_files
    #     if k in txt_files and path_to_id(wav_files[k]) in data.ID]

    data = data.merge(pd.DataFrame.from_records([{
            'path': wav_files[k],
            'sentence': txt_files[k].read_text().strip(),
            'ID': path_to_id(wav_files[k])
        } for k in tqdm(wav_files, desc='reading text') if k in txt_files]),
        how='right', on='ID')

    # create default hparams just for audio params
    hparams = create_hparams(hparams)
    stft = layers.TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax, hparams.use_mel).to(device)

    if whitelist_file:
        with open(whitelist_file, 'rb') as file:
            if whitelist_file.endswith('.pkl'):
                whitelist = pickle.load(file)
            else:
                whitelist = file.readlines()
        speakers = sorted(list(whitelist))
    elif single_speaker:
        prefix += '_'+single_speaker[:4]
        speakers = [single_speaker]
        data = data[data.ID==single_speaker]
    else:
        speakers = data.ID.unique()

    speaker_map = defaultdict(lambda: -1)
    speaker_map.update({s:i for i,s in enumerate(speakers)})

    data['speaker'] = data.ID.map(speaker_map)
    data = data[data.speaker>=0]

    # print(data.shape)

    print(f'{len(data)} utterances')
    print(f'{len(speaker_map)} speakers')

    is_val = pd.Series(index=data.index, dtype=np.bool)
    # stratified by speaker
    for _,g in data.groupby('speaker'):
        val_idxs = g.sample(
            int(np.ceil(val_size/len(speaker_map))), replace=False, random_state=0).index
        is_val[val_idxs] = True
    train_data, val_data = data[~is_val], data[is_val]

    def gen_spectra(data):
        for path in tqdm(data.path):
            parts = get_spectrum(
                stft, hparams, path, remove_noise=remove_noise)
            r = [parts['audio'], parts['spect']]
            yield path, r

    # save spectra with np.save
    if hparams.use_mel:
        spect_dir = f'spect_{hparams.n_mel_channels}_{int(hparams.mel_fmin)}_{int(hparams.mel_fmax)}'
    else:
        spect_dir = f'spect_lin_{hparams.filter_length}'
    if process_audio:
        for dir in (spect_dir, 'wav'):
            path = data_root.joinpath(dir)
            path.mkdir(exist_ok=True)

        for path, (wave, spect) in gen_spectra(data):
            if write_wav:
                np.save(data_root.joinpath('wav', path.stem), wave)
            np.save(data_root.joinpath(spect_dir, path.stem), spect)

    # write filelists
    for data, dest in (
            (train_data, f'filelists/{prefix}_train_filelist.txt'),
            (val_data, f'filelists/{prefix}_val_filelist.txt')):
        with open(dest, 'w') as fl:
            for path, text, speaker in zip(
                    tqdm(data.path, desc='writing '+dest),
                    data.sentence, data.speaker):
                fl.write(f'{data_root}/{spect_dir}/{path.stem}.npy|{text}|{speaker}|{0}\n')

    for data, dest in (
            (train_data, f'waveglow/{prefix}_train_filelist.txt'),
            (val_data, f'waveglow/{prefix}_val_filelist.txt')):
        with open(dest, 'w') as fl:
            for path in tqdm(data.path, desc='writing '+dest):
                fl.write(f'../{data_root}/wav/{path.stem}.npy\n')

    # store speaker ID mapping for later recovery
    with open(f'filelists/{prefix}_mappings.pkl', 'wb') as file:
        pickle.dump({
            'speaker': {s:i for s,i in speaker_map.items() if i>=0},
            # 'character': dict(char_freqs)
        }, file)

if __name__=='__main__':
    Fire(main)
