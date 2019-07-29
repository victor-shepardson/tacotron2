import sys, os, warnings
from multiprocessing.dummy import Pool as ThreadPool
from collections import defaultdict
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
        single_speaker=None, # use the single given client_id
        single_lang=None, # use the single given lang code (e.g. 'en')
        remove_noise=False, # spectral noise removal
        data_root='../data/mozilla_common_voice',
        prefix='mcv', # for output filenames
        whitelist_file=None,#'filelists/mcv_whitelist.pkl'#None
        val_size=100,
        min_speaker_samples=1,
        max_speakers_per_lang=None,
        hparams='',
        device='cpu',
        use_all_female=False,
        # threads=1 # soundfile seems not-threadsafe
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

    # create default hparams just for audio params
    hparams = create_hparams(hparams)
    stft = layers.TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax, hparams.use_mel).to(device)

    def gen_tables(fname):
        for i,l in enumerate(langs):
            data = pd.read_csv(os.path.join(data_root, l, fname), sep='\t')
            data['lang'] = l
            data['lang_idx'] = i
            yield data

    data = pd.concat(gen_tables('validated.tsv')).reset_index(drop=True)
    print(data.columns)

    if single_lang:
        prefix += '_'+single_lang

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
        data = data[data.client_id==single_speaker]
        data = data[data.lang==data.lang.value_counts().idxmax()]
    else:
        # convert client_id to speaker id and discard infrequent speakers
        # limit max speakers per lang so overrepresented langs dont cause underrepresented speakers with stratified sampling by lang
        f_ids = data[data.gender=='female'].client_id.unique()
        if use_all_female:
            print(f'including {len(f_ids)} client ids with female voices')
        speakers = np.unique([
            id for _, g in data.groupby('lang')
            for i, (id, count) in enumerate(g.client_id.value_counts().iteritems())
            if (count >= min_speaker_samples or (use_all_female and (id in f_ids)))
                and (max_speakers_per_lang is None or i < max_speakers_per_lang)
        ])

    speaker_map = defaultdict(lambda: -1)
    speaker_map.update({s:i for i,s in enumerate(speakers)})

    data['speaker'] = data.client_id.map(speaker_map)

    data = data[data.speaker>=0]

    # print(data.shape)

    print(f'{len(data)} utterances')
    print(f'{len(data.speaker.unique())} speakers')

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
    # char_freqs = defaultdict(int)
    # char_freqs_by_lang = {}
    # digraph_freqs_by_lang = {}
    # for lang, g in data.groupby('lang'):
    #     char_freqs_by_lang[lang] = defaultdict(int)
    #     digraph_freqs_by_lang[lang] = defaultdict(int)
    #     vc = char_vc(g.sentence, lang)
    #     for c,i in vc.iteritems():
    #         char_freqs_by_lang[lang][c] += i
    #         char_freqs[c] += i
    #     vc2 = char_vc(g.sentence, lang, ngram=2)
    #     for d,i in vc2.iteritems():
    #         digraph_freqs_by_lang[lang][d] += i

    def gen_spectra(data, include_raw=False):
        for fname, lang in zip(tqdm(data.path), data.lang):
            path = f'{data_root}/{lang}/clips/{fname}'
            if not path.endswith('.mp3'):
                path += '.mp3'
            parts = get_spectrum(
                stft, hparams, path, remove_noise=remove_noise)
            r = [parts['audio'], parts['spect']]
            if include_raw:
                r.append(parts['spect_raw'])
            yield fname, lang, r

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

        # for fname, lang, (w, s) in zip(tqdm(data.path), data.lang, gen_spectra(data)):
        for fname, lang, (w, s) in gen_spectra(data):
            if write_wav:
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
            # 'character': dict(char_freqs)
        }, file)

if __name__=='__main__':
    Fire(main)
