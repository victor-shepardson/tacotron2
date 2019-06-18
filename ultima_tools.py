import warnings, os
import numpy as np
from scipy import interpolate, signal
import torch
import torch.nn.functional as F
import librosa

def to_numpy(t):
    return t.data.cpu().numpy()

def to_torch(a, device, dtype=torch.float):
    return torch.tensor(a).to(device, dtype=dtype)

def partial_randperm(n, p=1, d=None):
    """
    permutation with variable locality and distance from range(n)
    not perfectly 'isotropic' but pretty good
    """
    d = n if d is None else d
    p *= d/(p+d) # correction for small d, large p related to increased prob. of later switch repeating earlier one
    idxs = np.arange(n)
    p_untouched = 1
    p_untouched_hist = [1 for _ in range(d)] # when d<n, have to keep running product
    for i in range(n-1):
        p_ = 1 + (p*(n-1)-n)/(n*p_untouched)
        b = int(np.random.rand()<p_)
        d_max = min(d, n-i-1)
        j = i+b*np.random.randint(1, d_max+1)
        idxs[[j, i]] = idxs[[i, j]]
        p_untouched_hist.append(1-p_/d_max)
        p_untouched *= p_untouched_hist[-1]
        p_untouched /= p_untouched_hist[-d-1]
    return idxs

def test_partial_randperm():
    import matplotlib.pyplot as plt
    n = 100
    p = 0.2
    d = 3
    x = np.arange(n)
    k = 10000
    p_expected = p*(1-1/n) * d/(p+d)
    d_expected = p*np.array([
        np.mean(np.abs(x-x_)[max(0, x_-d):min(n, x_+d+1)])
        for x_ in x])
    # blue: ratio of empirical/expected prob that an index is different
    p_emp = sum(
        (x != x[partial_randperm(n, p, d)]).astype(np.float)
        for _ in range(k))/k
    plt.plot(p_emp/p_expected)
    # orange: ratio of empirical/expected absolute difference in indices (large d)
    d_emp = sum(
        np.abs(x[partial_randperm(n, p, d)]-x).astype(np.float)
        for _ in range(k))/k
    plt.plot(d_emp/d_expected)


def formant_decompose(spect, axis=1):
    """separate log-spectrogram by quefrency with a linear-phase filter.

    operates on numpy arrays.
    """
    formants = signal.filtfilt(*signal.butter(8, 1/12), spect, axis=axis)
    return spect-formants, formants

def pitch_shift(spect, shift_pitch=0, shift_formant=0,
        interp_linear=True, mel_low=0, mel_high=8000):
    """formant-aware pitch shifting by resampling the mel spectrogram.

    `shift_pitch` and `shift_formant` given in semitones.
    `mel_low` and `mel_high` should match those used to produce `spect` originally.
    operates on numpy arrays.
    """
    mel_fs = librosa.mel_frequencies(spect.shape[1], mel_low, mel_high)
    if mel_low==0:
        mel_fs[0] = 1 # fudge 0 freq bin
    log_mel_fs = np.log2(mel_fs)

    p, f = formant_decompose(spect, axis=1)

    if interp_linear:
        p, f = np.exp(p), np.exp(f)

    p_fill_bin, f_fill_bin = (
        0 if shift >=0 else -1 for shift in (shift_pitch, shift_formant))
    p = interpolate.interp1d(
        log_mel_fs+shift_pitch/12, p,
        axis=1, fill_value=p[:, p_fill_bin, :], bounds_error=False
        )(log_mel_fs)
    f = interpolate.interp1d(
        log_mel_fs+shift_formant/12, f,
        axis=1, fill_value=f[:, f_fill_bin, :], bounds_error=False
        )(log_mel_fs)

    if interp_linear:
        p, f = np.log(p), np.log(f)

    return f+p

def time_stretch(spect, factor):
    """change rate of spectrogram by linear interpolation.

    `factor` is stretch factor, i.e. 0.5 means double speed.
    negative `factor` will reverse time. operates on torch tensors.
    """
    if factor<0:
        spect = torch.flip(spect, -1)
        factor *= -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return F.interpolate(spect, scale_factor=factor, mode='linear')

def load_text(filename):
    if os.path.isdir(filename):
        filenames = [f for f in os.listdir(filename) if os.path.isfile(f)]
        if len(filenames)==0:
            raise ValueError('textfile directory contains no files')
        filename = np.random.choice(filenames)
    with open(filename, 'r') as file:
        return file.read()

def sample_text(text, lines, words, chars):
    if lines is not None:
        text = '\n'.join(sample_chunks(text.splitlines(), lines))
    if words is not None:
        text = ' '.join(sample_chunks(text.split(), words))
    if chars is not None:
        text = ''.join(sample_chunks(text, chars))
    return text

def sample_chunks(chunks, n):
    stride = 1
    if n<0:
        stride = -1
        n *= -1
    n = min(len(chunks), n)
    start = np.random.randint(len(chunks)-n) if n<len(chunks) else 0
    chunks = chunks[start:start+n]
    return chunks[::stride]

def mel_inv(spect, hparams):
    """transform the log-mel spectrogram to linear STFT"""
    mel_fs = librosa.mel_frequencies(
        spect.shape[1], hparams.mel_fmin, hparams.mel_fmax)
    spect = np.exp(spect)
    spect = interpolate.interp1d(
        mel_fs, spect, axis=1, fill_value=spect[:, -1, :], bounds_error=False
    )(np.linspace(0, hparams.sampling_rate/2, hparams.filter_length//2+3)[1:-1])
    return spect
