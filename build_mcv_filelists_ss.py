import sys, os, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from fire import Fire

import layers
from hparams import create_hparams
from utils import load_audio_to_torch
from text.cleaners import multi_cleaners

"""Preprocess audio and build filelists for both tacotron2 and waveglow.
Assumes waveglow is nested in tacotron2 directory and not the other way around."""

# debug = bool(sys.argv[2]) if len(sys.argv)>2 else False

def main(
        process_audio=False,
        single_speaker=None,
        single_lang=None,
        remove_noise=False,
        data_root='../data/mozilla_common_voice',
        prefix='mcv', # for output filenames
        whitelist_file=None,#'filelists/mcv_whitelist.pkl'#None
        val_size=100,
        hparams=''
    ):
    langs = [
        d for d in os.listdir(data_root)
        if not d.startswith('.')
        and os.path.exists(os.path.join(data_root, d, 'clips'))]
    print(f'found {len(langs)} languages: {langs}')
    if single_lang is not None:
        assert single_lang in langs
        print(f'using {single_lang} only')
        langs = [single_lang]
    min_speaker_samples = 100
    max_speakers_per_lang = 16

    # create default hparams just for audio params
    hparams = create_hparams(hparams)
    stft = layers.TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax, hparams.use_mel)

    def gen_tables(fname):
        for i,l in enumerate(langs):
            data = pd.read_csv(os.path.join(data_root, l, fname), sep='\t')
            data['lang'] = l
            data['lang_idx'] = i
            yield data

    data = pd.concat(gen_tables('validated.tsv')).reset_index(drop=True)

    if whitelist_file:
        with open(whitelist_file, 'rb') as file:
            whitelist = pickle.load(file)
        speakers = sorted(list(whitelist))
    elif single_speaker:
        prefix += '_'+single_speaker[:4]
        speakers = [single_speaker]
        data = data[data.client_id==single_speaker]
        data = data[data.lang==data.lang.value_counts().idxmax()]
    else:
        # convert client_id to speaker id and discard infrequent speakers
        # limit max speakers per lang so overrepresented langs dont cause underrepresented speakers with stratified sampling by lang
        speakers = np.unique([
            id for _, g in data.groupby('lang')
            for i, (id, count) in enumerate(g.client_id.value_counts().iteritems())
            if count >= min_speaker_samples and i < max_speakers_per_lang
        ])

    speaker_map = defaultdict(lambda: -1)
    speaker_map.update({s:i for i,s in enumerate(speakers)})

    data['speaker'] = data.client_id.map(speaker_map)

    data = data[data.speaker>=0]

    print(data.shape)

    # if debug:
    #     data = data.sample(200)
    #     val_size = len(langs)

    # original validation split tests generalization across speakers (for recognition)
    # for multi-speaker TTS model, we want to test across sentences within speakers
    is_val = pd.Series(index=data.index, dtype=np.bool)
    # stratified by language
    for _,g in data.groupby('lang'):
        val_idxs = g.sample(val_size, replace=False, random_state=0).index
        is_val[val_idxs] = True
    train_data, val_data = data[~is_val], data[is_val]

    def char_vc(s, lang, clean=multi_cleaners, ngram=1):
        s = ''.join(s)
        if clean is not None:
            s = clean(s, {'lang': lang})
        else:
            s = s.lower()
        grams = [s[i:i+ngram] for i in range(len(s)-ngram+1)]
        return pd.Series(grams).value_counts()

    # compute character frequencies after cleaning
    char_freqs = defaultdict(int)
    char_freqs_by_lang = {}
    digraph_freqs_by_lang = {}
    for lang, g in data.groupby('lang'):
        char_freqs_by_lang[lang] = defaultdict(int)
        digraph_freqs_by_lang[lang] = defaultdict(int)
        vc = char_vc(g.sentence, lang)
        for c,i in vc.iteritems():
            char_freqs_by_lang[lang][c] += i
            char_freqs[c] += i
        vc2 = char_vc(g.sentence, lang, ngram=2)
        for d,i in vc2.iteritems():
            digraph_freqs_by_lang[lang][d] += i

    def gen_spectra(data, include_raw=False):
        for fname, lang in zip(data.path, data.lang):
            path = f'{data_root}/{lang}/clips/{fname}.mp3'
            audio = load_audio_to_torch(path, hparams.sampling_rate, wav_scale=False)[0]
            spect = spect_raw = stft.mel_spectrogram(audio.unsqueeze(0)).squeeze(0).numpy()

            if spect.shape[-1] < 30:
                warnings.warn(f'unexpectedly short audio: {path}')

            drop_lf_bands = 3 #ignore noisy low frequency bands when detecting silence
            peak_range = 3 # range below overall spectral peak for a frame to be considered speech
            trim = (1, 3) # include frames before, after first/last detected speech
            noise_quant = (0.03, 0.1) # mean frame intensity quantile to use as noise
            noise_reduce = 0.7 # fraction of noise to replace with noise_floor
            noise_floor = 5e-5

            # trim leading/trailing silence
            spectral_peaks = np.max(spect[drop_lf_bands:], axis=0)
            loud = np.argwhere(
                (spectral_peaks > np.max(spectral_peaks)-peak_range)
            ).squeeze()
            lo, hi = max(0, loud[0]-trim[0]), min(spect.shape[1], loud[-1]+trim[1])

            # reduce background noise
            noise = 0
            if remove_noise:
                spectral_mean = np.mean(spect[drop_lf_bands:], axis=0)
                quiet = np.argwhere((
                    (spectral_mean < np.quantile(spectral_mean, noise_quant[1]))
                    & (spectral_mean > np.quantile(spectral_mean, noise_quant[0]))
                )).squeeze()
                if quiet.ndim > 0 and len(quiet) > 0:
                    noise = spect[:, quiet].mean(1, keepdims=True)

            spect = spect[:, lo:hi]

            if remove_noise:
                spect = np.log(np.maximum(
                    np.exp(spect) - noise_reduce*np.exp(noise),
                    noise_floor))

            r = [
                audio[lo*hparams.hop_length:hi*hparams.hop_length],
                spect
            ]
            if include_raw:
                r.append(spect_raw)
            yield r

    # save spectra with np.save
    if hparams.use_mel:
        spect_dir = f'spect_{hparams.n_mel_channels}_{int(hparams.mel_fmin)}_{int(hparams.mel_fmax)}'
    else:
        spect_dir = f'spect_lin_{hparams.filter_length}'
    if process_audio:
        for lang in langs:
            for dir in (spect_dir, 'wav'):
                path = os.path.join(data_root, lang, dir)
                if not os.path.exists(path):
                    os.mkdir(path)

        for fname, lang, (w, s) in zip(tqdm(data.path), data.lang, gen_spectra(data)):
            np.save(f'{data_root}/{lang}/wav/{fname}', w)
            np.save(f'{data_root}/{lang}/{spect_dir}/{fname}', s)

    # write filelists
    for data, dest in (
            (train_data, f'filelists/{prefix}_train_filelist.txt'),
            (val_data, f'filelists/{prefix}_val_filelist.txt')):
        with open(dest, 'w') as fl:
            for fname, text, speaker, lang, lang_idx in zip(
                    tqdm(data.path, desc='writing '+dest),
                    data.sentence, data.speaker, data.lang, data.lang_idx):
                fl.write(f'{data_root}/{lang}/{spect_dir}/{fname}.npy|{text}|{speaker}|{lang_idx}\n')

    for data, dest in (
            (train_data, f'waveglow/{prefix}_train_filelist.txt'),
            (val_data, f'waveglow/{prefix}_val_filelist.txt')):
        with open(dest, 'w') as fl:
            for fname, lang, in zip(
                    tqdm(data.path, desc='writing '+dest), data.lang):
                fl.write(f'../{data_root}/{lang}/wav/{fname}.npy\n')

    # store client_id and language code mappings for later recovery
    with open(f'filelists/{prefix}_mappings.pkl', 'wb') as file:
        pickle.dump({
            'language': {l:i for i,l in enumerate(langs)},
            'speaker': {s:i for s,i in speaker_map.items() if i>=0},
            'character': dict(char_freqs)
        }, file)

if __name__=='__main__':
    Fire(main)
