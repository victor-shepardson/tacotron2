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

def main(
        process_audio=False, # precompute spectra
        write_wav=False, # also store the trimmed raw audio as .npy
        remove_noise=False, # spectral noise removal
        data_root='../data/LJSpeech-1.1',
        prefix='ljs', # for output filenames
        val_size=100,
        hparams='',
        device='cpu',
        debug=False
        # threads=1 # soundfile seems not-threadsafe
    ):
    data_root = Path(data_root)#.expanduser().resolve()

    data = pd.read_csv(
        data_root.joinpath('metadata.csv'), sep='|', names=['fname', 'raw_text', 'sentence'])

    wav_files = {p.stem:p
        for p in tqdm(data_root.joinpath('wavs').rglob('*.wav'), desc='wav')}

    data = data.merge(pd.DataFrame.from_records([{
            'fname': k,
            'path': wav_files[k]
        } for k in tqdm(wav_files, desc='associating paths')]),
        how='left', on='fname')

    # create default hparams just for audio params
    hparams = create_hparams(hparams)
    stft = layers.TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax, hparams.use_mel).to(device)

    data['speaker'] = 0

    # print(data.shape)

    if debug:
        data = data.iloc[:200]

    print(f'{len(data)} utterances')

    is_val = pd.Series(index=data.index, dtype=np.bool)
    val_idxs = data.sample(
        val_size, replace=False, random_state=0).index
    is_val[val_idxs] = True
    train_data, val_data = data[~is_val], data[is_val]

    def gen_spectra(data):
        for path in tqdm(data.path):
            parts = get_spectrum(
                stft, hparams, path, remove_noise=remove_noise, trim=False)
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


if __name__=='__main__':
    Fire(main)
