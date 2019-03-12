# to create executable:
# > conda activate waveglow
# > source bundle.fish

# TODO:
# - multichannel: allow branching at various places
# - performance: how hard to get cpu inference faster?
# - set decoder steps manually when shuffling code

import sys
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
from librosa.output import write_wav

import fire

import ultima_tools as ut
from ultima_tools import to_torch, to_numpy

sys.path.append('waveglow/')

def main(text,
        shift_pitch=0, shift_formant=0, # semitones
        channels=1,
        stretch_time=1, # negative to reverse time; large values slow to render
        shuffle_text=0, shuffle_code=0, # 0 to 1
        decoder_steps=None, # if using shuffle_code, will tend to run to max steps
        outfile='out.wav', verbose=True
    ):
    use_gpu=False
    glow_temperature=0.666

    if stretch_time==0:
        warnings.warn('stretch_time cannot be zero')
    if shuffle_text<0 or shuffle_code<0 or shuffle_text>1 or shuffle_code>1:
        warnings.warn('shuffle parameters should be between 0 and 1')

    # #### Setup hparams

    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.gpu = use_gpu
    if decoder_steps is not None:
        hparams.max_decoder_steps = decoder_steps

    device = torch.device('cuda') if hparams.gpu else torch.device('cpu')

    # #### Load tacotron model

    checkpoint_path = "tacotron2_statedict.pt"
    tacotron = load_model(hparams)
    load_kwargs = {} if hparams.gpu else {'map_location':'cpu'}
    tacotron.load_state_dict(
        torch.load(checkpoint_path, **load_kwargs)['state_dict'])
    tacotron.eval()

    # #### Load WaveGlow model

    waveglow_path = 'waveglow_old.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        waveglow = torch.load(waveglow_path, **load_kwargs)['model']
    waveglow.to(device)
    waveglow.eval()

    # #### Prepare text input

    # shuffle text
    text_shuf_amt = shuffle_text
    text_shuf_dist = int((shuffle_text+1)**np.log2(len(text)))
    text = ''.join(np.array([t for t in text])[
        ut.partial_randperm(len(text), text_shuf_amt, text_shuf_dist)])
    if verbose:
        print(text)

    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = to_torch(sequence, device, torch.long)

    # #### Pass text through tacotron
    if verbose:
        print('tacotron encoder...')
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
    if verbose:
        print(f'writing "{outfile}"')
    write_wav(outfile, to_numpy(audio), hparams.sampling_rate)

if __name__=='__main__':
    fire.Fire(main)
