# to create executable:
# > conda activate waveglow
# > source bundle.fish

# TODO:
# - multichannel: test pysoundfile
# - weights: test path params
# - performance: how hard to get cpu inference faster?

import sys, os
import multiprocessing
import warnings

# pyinstaller fix -- needs to be before one of these imports apparently
multiprocessing.freeze_support()

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from train import load_model
from text import text_to_sequence
# from librosa.output import write_wav
import soundfile

import fire

import ultima_tools as ut
from ultima_tools import to_torch, to_numpy

sys.path.append('waveglow/')

def main(text='', textfile=None, lines=None, words=None, chars=None,
        shift_pitch=0, shift_formant=0, stretch_time=1,
        shuffle_text=0, shuffle_code=0,
        channels=1, decoder_steps=None,
        model_dir='.',
        tacotron_weights='tacotron2_statedict.pt',
        waveglow_weights='waveglow_old.pt',
        outfile='out.wav', verbose=True
    ):
    """generate audio from text using tacotron2 + waveglow.

    Based on NVIDIA's open source implementations and using their checkpoints
    pretrained on the LJ Speech dataset.

    Args:
        text (str): text to vocalize. Either `text` or `textfile` must be supplied.
            If both are supplied, the texts will be concatenated.
        textfile (str): path to read text from; either a file or a directory from
            which a random file will be chosen.
        lines/words/chars (int): number of lines/words/characters to read from
            a random starting point in the input text. lines, words and chars
            will be applied in that order if more than one is supplied.
            If none are supplied, whole text is used.
        shift_pitch (float): pitch shift in semitones.
        shift_formant (float): formant shift in semitones.
        channels (int): number of channels. When using `decoder_steps`,
            channels diverge before attention layer (different timing/content).
            Otherwise, they just get different waveglow 'z' (subtle stereo effect).
        stretch_time (float): stretch factor, i.e. 2 means output is twice as long.
            Negative values reverse time. Values > 1 increase processing time.
        shuffle_text (float): input character-shuffling intensity between 0 and 1.
        shuffle_code (float): preattention code-shuffling intensity between 0 and 1.
            Using `shuffle_code` tends to confuse tacotron so it will run to the
            maximum allowed decoder steps (by default, 1000 or ~11 seconds).
            It is recommended to set `decoder_steps` when using `shuffle_code` or `channels`.
        decoder_steps (int): number of spectral frames to render.
            Each frame is about 11 ms. If not provided, tacotron will attempt to
            terminate itself when the text is finished.
        model_dir (str): root directory for `tacotron_path` and `waveglow_path`.
        tacotron_path (str): path to tacotron model weights (relative to model_dir).
        waveglow_path (str): path to waveglow model weights (relative to model_dir).
        outfile (str): file name to write output in wav format.
        verbose (bool): whether to print progress messages while running.
    """
    use_gpu=False
    glow_temperature=0.666

    if text is None and textfile is None:
        raise ValueError('must supply either text or textfile')
    if stretch_time==0:
        raise ValueError('stretch_time cannot be zero')
    if shuffle_text<0 or shuffle_code<0 or shuffle_text>1 or shuffle_code>1:
        raise ValueError('shuffle parameters should be between 0 and 1')

    # #### Prepare text input

    if textfile is not None:
        text = ''.join((text, ut.load_text(textfile)))
    text = ut.sample_text(text, lines, words, chars)
    # shuffle
    text_shuf_amt = shuffle_text
    text_shuf_dist = int((shuffle_text+1)**np.log2(len(text)))
    text = ''.join(np.array([t for t in text])[
        ut.partial_randperm(len(text), text_shuf_amt, text_shuf_dist)])

    if verbose:
        print(text)

    # #### Setup hparams, gpu

    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.gpu = use_gpu
    if decoder_steps is not None:
        hparams.max_decoder_steps = decoder_steps

    device = torch.device('cuda') if hparams.gpu else torch.device('cpu')

    # #### Load tacotron model

    print('loading model parameters...')

    load_kwargs = {} if hparams.gpu else {'map_location':'cpu'}

    tacotron_path = os.path.join(model_dir, tacotron_weights)
    tacotron = load_model(hparams)
    load_kwargs = {} if hparams.gpu else {'map_location':'cpu'}
    tacotron.load_state_dict(
        torch.load(tacotron_path, **load_kwargs)['state_dict'])
    tacotron.eval()

    # #### Load WaveGlow model

    waveglow_path = os.path.join(model_dir, waveglow_weights)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        waveglow = torch.load(waveglow_path, **load_kwargs)['model']
    waveglow.to(device)
    waveglow.eval()

    # #### Pass text through tacotron

    if verbose:
        print('tacotron encoder...')

    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = to_torch(sequence, device, torch.long)
    with torch.no_grad():
        encoded = tacotron.encode(sequence)
    # shuffle encoding
    code_shuf_amt = shuffle_code
    code_shuf_dist = int((shuffle_code+1)**np.log2(encoded.size(1)))
    encoded = encoded[:,
        ut.partial_randperm(encoded.size(1), code_shuf_amt, code_shuf_dist)]

    if verbose:
        print('tacotron decoder...')
    if decoder_steps is not None:
        encoded = encoded.expand(channels, -1, -1)
    with torch.no_grad():
        _, spect, _, _ = tacotron.decode(
            encoded, use_gate=(decoder_steps is None))

    # #### pitch and time modulation

    # pitch shift
    if shift_pitch!=0 or shift_formant!=0:
        spect = to_torch(ut.pitch_shift(
            to_numpy(spect), shift_pitch, shift_formant
            ), device)
    # time stretch
    spect = ut.time_stretch(spect, stretch_time)
    # pass again through postnet
    if shift_pitch!=0 or shift_formant!=0 or np.abs(stretch_time)>1:
        with torch.no_grad():
            spect = tacotron.apply_postnet(spect)

    # #### Synthesize audio from spectrogram using WaveGlow

    if verbose:
        print('waveglow vocoder...')
    if decoder_steps is None:
        spect = spect.expand(channels, -1, -1)
    with torch.no_grad():
        audio = waveglow.infer(spect, sigma=glow_temperature, verbose=True)

    # #### write to wav file

    peak = audio.abs().max()
    if peak>=0.95:
        warnings.warn(f'normalizing audio with peak {peak}')
        audio *= 0.95/peak
    if verbose:
        print(f'writing "{outfile}"')
    audio = to_numpy(audio)

    # if audio.shape[0]==1:
        # audio = audio[0]
    # write_wav(outfile, audio, hparams.sampling_rate)
    soundfile.write(outfile, audio.T, hparams.sampling_rate, format='WAV')

if __name__=='__main__':
    fire.Fire(main)
