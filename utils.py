import os
from pathlib import Path
import numpy as np
from scipy.io.wavfile import read as wavread
from librosa.core import load as audioread
import torch

MAX_WAV_VALUE = 32768.0

def get_mask_from_lengths(lengths, device=None):
    max_len = torch.max(lengths).item()
    if device is None:
        # ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
        ids = torch.arange(0, max_len, dtype=torch.long, device=lengths.device)
    else:
        ids = torch.arange(0, max_len, dtype=torch.long, device=device)
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path):
    sampling_rate, data = wavread(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_audio_to_torch(full_path, sampling_rate, limit=True, wav_scale=True):
    """
    Loads audio data into torch array
    """
    # try:
    #     file_sampling_rate, data = wavread(full_path)
    #     assert sampling_rate==file_sampling_rate
    # except Exception:
    data, _ = audioread(full_path, sr=sampling_rate, mono=True, res_type='kaiser_fast')
    if limit:
        data /= max(1, np.max(np.abs(data)))
    if wav_scale:
        data *= MAX_WAV_VALUE
    return torch.from_numpy(data).float(), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    filepaths_and_text = []
    if isinstance(filenames, str) or isinstance(filenames, Path):
        filenames = [filenames]
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            lines = [line.strip().split(split) for line in f]
        filepaths_and_text.extend([
            [os.path.expanduser(hd)]+tl for hd, *tl in lines])
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
