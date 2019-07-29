import torch
import numpy as np
from scipy.signal import get_window
import librosa.util as librosa_util
from utils import load_audio_to_torch

def get_spectrum(stft, hparams, path, device='cpu',
        drop_lf_bands=3, #ignore noisy low frequency bands when detecting silence
        peak_range=3, # range below overall spectral peak for a frame to be considered speech
        trim=(1, 3), # include frames before, after first/last detected speech
        noise_quant=(0.03, 0.1), # mean frame intensity quantile to use as noise
        noise_reduce=0.7, # fraction of noise to replace with noise_floor
        noise_floor=5e-5,
        remove_noise=False):
    audio = load_audio_to_torch(
        path, hparams.sampling_rate, wav_scale=False)[0]
    spect = spect_raw = stft.mel_spectrogram(
        audio.to(device).unsqueeze(0)).squeeze(0).cpu().numpy()

    if spect.shape[-1] < 30:
        warnings.warn(f'unexpectedly short audio: {path}')

    # trim leading/trailing silence
    if trim is not None and trim!=False:
        spectral_peaks = np.max(spect[drop_lf_bands:], axis=0)
        loud = np.argwhere(
            (spectral_peaks > np.max(spectral_peaks)-peak_range)
        ).squeeze()
        lo, hi = max(0, loud[0]-trim[0]), min(spect.shape[1], loud[-1]+trim[1])
    else:
        lo, hi = 0, spect.shape[1]

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

    return {
        'audio': audio[lo*hparams.hop_length:hi*hparams.hop_length],
        'spect': spect,
        'spect_raw': spect_raw
    }

def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x


def griffin_lim(magnitudes, stft_fn, n_iters=30, verbose=False):
    """
    PARAMS
    ------
    magnitudes: spectrogram magnitudes
    stft_fn: STFT class with transform (STFT) and inverse (ISTFT) methods
    """

    angles = np.angle(np.exp(2j * np.pi * np.random.rand(*magnitudes.size())))
    angles = angles.astype(np.float32)
    angles = torch.autograd.Variable(torch.from_numpy(angles))
    signal = stft_fn.inverse(magnitudes, angles).squeeze(1)

    iters = range(n_iters)
    if verbose:
        from tqdm import tqdm
        iters = tqdm(list(iters), desc='GL step')

    for i in iters:
        _, angles = stft_fn.transform(signal)
        signal = stft_fn.inverse(magnitudes, angles).squeeze(1)
    return signal


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C
