import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression
from stft import STFT


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_spect_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0, use_mel=True):
        super(TacotronSTFT, self).__init__()
        self.use_mel = use_mel
        self.n_spect_channels = n_spect_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_spect_channels, mel_fmin, mel_fmax)
        # inv_mel_basis = np.maximum(np.linalg.pinv(mel_basis), 0)
        inv_mel_basis = (mel_basis / np.maximum(mel_basis.sum(0), 1e-3)).T / np.maximum(mel_basis.sum(1), 1e-3)
        mel_basis = torch.from_numpy(mel_basis).float()
        inv_mel_basis = torch.from_numpy(inv_mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('inv_mel_basis', inv_mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_spect_channels, T)
        """
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output

    def mel_inv(self, y):
        lin_output = self.spectral_de_normalize(y)
        lin_output = torch.matmul(self.inv_mel_basis, lin_output).clamp(1e-5)
        return lin_output

    def loglin_spectrogram(self, y):
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        return self.spectral_normalize(magnitudes)

    def loglin_inv(self, y):
        return self.spectral_de_normalize(y)

    def spectrogram(self, y):
        return self.mel_spectrogram(y) if self.use_mel else self.loglin_spectrogram(y)

    def inv_spectrogram(self, y):
        return self.mel_inv(y) if self.use_mel else self.loglin_inv(y)
