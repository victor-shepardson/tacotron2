import random
import numpy as np
import pandas as pd
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_audio_to_torch, load_filepaths_and_text
from text import text_to_sequence

class StratifiedSampler(torch.utils.data.Sampler):
    def __init__(self, var):
        self.var = pd.Series(var)
        self.vc = self.var.value_counts()

    def __iter__(self):
        def gen_samples():
            d = {c: np.random.permutation(g.index)
                 for c,g in self.var.groupby(self.var)}
            for k in range(self.vc.min()):
                for c in np.random.permutation(list(d)):
                    yield d[c][k]
        return gen_samples()

    def __len__(self):
        # one epoch is smallest class * number of classes
        return len(self.vc)*self.vc.min()

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        # random.shuffle(self.audiopaths_and_text)

    def get_data_tuple(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[:2]
        text = self.get_text(*r)
        mel = self.get_mel(audiopath)
        r = [text, mel]
        r += audiopath_and_text[2:]
        # if len(audiopath_and_text) > 2:
        #     speaker = audiopath_and_text[2]
        #     r.append(speaker)
        # if len(audiopath_and_text) > 3:
        #     language = audiopath_and_text[3]
        #     r.append(language)
        return r

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            if '.wav' in filename:
                audio, sampling_rate = load_wav_to_torch(filename)
            else:
                audio, sampling_rate = load_audio_to_torch(filename, self.stft.sampling_rate)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} {} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return melspec

    def get_text(self, audio_path, text, speaker=None, lang=None):
        text_norm = torch.IntTensor(text_to_sequence(
            text, self.text_cleaners, {'lang':lang}))
        return text_norm

    def __getitem__(self, index):
        return self.get_data_tuple(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized] (+speaker, language)
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        r = [text_padded, input_lengths, mel_padded, gate_padded, output_lengths]

        if len(batch[0]) > 2:
            speaker = torch.LongTensor(len(batch))
            for i in range(len(ids_sorted_decreasing)):
                spk = int(batch[ids_sorted_decreasing[i]][2])
                speaker[i] = spk
            r.append(speaker)

        if len(batch[0]) > 3:
            language = torch.LongTensor(len(batch))
            for i in range(len(ids_sorted_decreasing)):
                lang = int(batch[ids_sorted_decreasing[i]][3])
                language[i] = lang
            r.append(language)

        return r
