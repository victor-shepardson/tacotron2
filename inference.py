# to create executable:
# > conda activate waveglow
# > source bundle.fish

# TODO:
# - performance: how hard to get cpu inference faster?
# - preview: implement faster (e.g. griffin-lim) vocoder?
# - random seed arg?

import sys, os
import multiprocessing
import warnings

# pyinstaller fix -- needs to be before one of the subsequent imports apparently
multiprocessing.freeze_support()

import numpy as np
import torch
import soundfile
import fire
from scipy import interpolate
import librosa

from hparams import create_hparams
from text import text_to_sequence, sequence_to_text
from stft import STFT
from audio_processing import griffin_lim
from layers import TacotronSTFT

import ultima_tools as ut
from ultima_tools import to_torch, to_numpy

sys.path.insert(0, 'waveglow/')

def main(text, textfile=None, lines=None, words=None, chars=None,
        shift_pitch=0, shift_formant=0, stretch_time=1,
        shuffle_text=0, shuffle_code=0,
        channels=1, decoder_steps=None,
        speaker_id=None,
        draft=False,
        model_dir='.',
        model='nvidia_lj',
        tacotron_file=None, waveglow_file=None,
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
        speaker_id (int): index of speaker (only if using multi-speaker model)
        draft (bool): Use fast Griffin-Lim vocoder instead of WaveGlow.
        model (str): select which model to use.
            single speaker models: 'nvidia_lj', 'mcv_6506', 'mcv_9ff9', 'mcv_c49c'.
            multi speaker models: 'mcv_8_97'.
        model_dir (str): root directory for `tacotron_path` and `waveglow_path`.
        tacotron_file (str): override path to tacotron weights (relative to model_dir).
        waveglow_file (str): override path to waveglow weights (relative to model_dir).
        outfile (str): file name to write output in wav format.
        verbose (bool): whether to print progress messages while running.
    """
    use_gpu = False
    glow_temperature = 0.666
    ss_models = ['nvidia_lj', 'mcv_6506', 'mcv_c49c', 'mcv_9ff9']
    ms_models = ['mcv_8_97']

    if text is None and textfile is None:
        raise ValueError('must supply either text or textfile')
    if stretch_time==0:
        raise ValueError('stretch_time cannot be zero')
    if shuffle_text<0 or shuffle_code<0 or shuffle_text>1 or shuffle_code>1:
        raise ValueError('shuffle parameters should be between 0 and 1')
    if model not in ss_models+ms_models:
        raise ValueError(f"""model must be one of:
            {ss_models} (single speaker),
            or {ms_models} (multi speaker)""")

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

    # torch.set_num_threads(os.cpu_count()-1)

    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.gpu = use_gpu
    if decoder_steps is not None:
        hparams.max_decoder_steps = decoder_steps

    device = torch.device('cuda') if hparams.gpu else torch.device('cpu')

    print('loading model parameters...')
    load_kwargs = {} if hparams.gpu else {'map_location':'cpu'}

    # #### Load tacotron model
    metadata = {}
    if model=='nvidia_lj':
        from model import Tacotron2
        tacotron_file = tacotron_file or 'tacotron2_statedict.pt'
        waveglow_file = waveglow_file or 'waveglow_256channels.pt'
    elif model=='mcv_6506':
        from model import Tacotron2
        tacotron_file = tacotron_file or 'tacotron2_mcv_6506.pt'
        waveglow_file = waveglow_file or 'waveglow_256channels.pt'
        hparams.text_cleaners = ['multi_cleaners']
        metadata = {'lang': 'tr'}
    elif model=='mcv_c49c':
        from model import Tacotron2
        tacotron_file = tacotron_file or 'tacotron2_mcv_c49c.pt'
        waveglow_file = waveglow_file or 'waveglow_256channels.pt'
        hparams.text_cleaners = ['multi_cleaners']
        metadata = {'lang': 'eo'}
    elif model=='mcv_9ff9':
        from model import Tacotron2
        tacotron_file = tacotron_file or 'tacotron2_mcv_9ff9.pt'
        waveglow_file = waveglow_file or 'waveglow_256channels.pt'
        hparams.text_cleaners = ['multi_cleaners']
        metadata = {'lang': 'cy'}
    elif model=='mcv_8_97':
        from model_cond import Tacotron2
        tacotron_file = tacotron_file or 'tacotron2_mcv_8_97.pt'
        # this is messy as hell. should use whole model pickling
        hparams.n_speakers = 97
        hparams.speaker_embedding_dim = 32
        hparams.n_languages = 8
        hparams.language_embedding_dim = 32
        hparams.symbols_embedding_dim = 448
        hparams.encoder_n_convolutions = 4
        hparams.text_cleaners = ['transliteration_cleaners']
        waveglow_file = waveglow_file or 'waveglow_mcv.pt'
    else:
        raise ValueError('unknown model')

    tacotron = Tacotron2(hparams)
    tacotron_path = os.path.join(model_dir, tacotron_file)
    tacotron.load_state_dict(
        torch.load(tacotron_path, **load_kwargs)['state_dict'])
    tacotron.to(device)
    tacotron.eval()

    # #### Load WaveGlow model
    if not draft:
        waveglow_path = os.path.join(model_dir, waveglow_file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waveglow = torch.load(waveglow_path, **load_kwargs)['model']
        waveglow.to(device)
        waveglow.eval()

    # #### Pass text through tacotron

    if verbose:
        print('tacotron inference...')

    sequence = np.array(text_to_sequence(
        text, hparams.text_cleaners, metadata
        ))[None, :]
    if verbose:
        print(f'normalized text: "{sequence_to_text(sequence[0])}"')
    sequence = to_torch(sequence, device, torch.long)
    seq_len = sequence.shape[1]

    # shuffle encoding
    code_shuf_amt = shuffle_code
    code_shuf_dist = int((shuffle_code+1)**np.log2(seq_len))
    code_perm = ut.partial_randperm(seq_len, code_shuf_amt, code_shuf_dist)

    if model in ss_models:
        with torch.no_grad():
            encoded = tacotron.encode(sequence)

        encoded = encoded[:, code_perm]

        if decoder_steps is not None:
            encoded = encoded.expand(channels, -1, -1)
        with torch.no_grad():
            _, spect, _, _ = tacotron.decode(
                encoded, use_gate=(decoder_steps is None))

    elif model=='mcv_8_97':
        whitelist = [
            0, 2, 3, 4, 8, 10, 14, 16, 20, 22, 26, 30, 31, 36, 37, 39, 44, 46,
            55, 58, 60, 61, 66, 67, 70, 77, 85, 87, 90, 91, 94, 96, 97]
        pairs = np.array(tacotron.get_speaker_lang_pairs())[whitelist]
        if speaker_id is None:
            speaker_id = np.random.choice(len(pairs))
        if speaker_id >= len(pairs):
            raise ValueError(f'speaker_id must be from 0 to {len(pairs)-1}')
        speaker, lang = pairs[speaker_id]
        if verbose:
            print(f'speaker_id {speaker_id}: speaker {speaker}, language {lang}')
        spk_emb = tacotron.speaker_embedding(torch.LongTensor(
            [speaker], device=device))
        lang_emb = tacotron.language_embedding(torch.LongTensor(
            [lang], device=device))

        with torch.no_grad():
            encoded = tacotron.encode(sequence, lang_emb)

        encoded = encoded[:, code_perm]

        if decoder_steps is not None:
            encoded = encoded.expand(channels, -1, -1)
            lang_emb = lang_emb.expand(channels, -1)
            spk_emb = spk_emb.expand(channels, -1)
        with torch.no_grad():
            _, spect, _, _ = tacotron.decode(
                encoded, spk_emb, lang_emb, use_gate=(decoder_steps is None))


    # #### pitch and time modulation

    # pitch shift
    if shift_pitch!=0 or shift_formant!=0:
        spect = to_torch(ut.pitch_shift(
            to_numpy(spect), shift_pitch, shift_formant,
            mel_low=hparams.mel_fmin, mel_high=hparams.mel_fmax
            ), device)
    # time stretch
    spect = ut.time_stretch(spect, stretch_time)
    # pass again through postnet
    if shift_pitch!=0 or shift_formant!=0 or np.abs(stretch_time)>1:
        with torch.no_grad():
            spect = tacotron.apply_postnet(spect)

    # #### Synthesize audio from spectrogram using WaveGlow

    if verbose:
        print(f'{"griffin-lim" if draft else "waveglow"} vocoder...')
    if decoder_steps is None:
        spect = spect.expand(channels, -1, -1)

    if draft:
        audio = 20*griffin_lim_synth(spect, hparams)
    else:
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

    soundfile.write(outfile, audio.T, hparams.sampling_rate, format='WAV')


# def griffin_lim_synth(spect, hparams, n_iters=30):
#     S = STFT(filter_length=hparams.filter_length, win_length=hparams.win_length, hop_length=hparams.hop_length)
#     mel_fs = librosa.mel_frequencies(
#         spect.shape[1], hparams.mel_fmin, hparams.mel_fmax)
#     spect = np.exp(spect)
#     spect = interpolate.interp1d(
#         mel_fs, spect, axis=1, fill_value=spect[:, -1, :], bounds_error=False
#     )(np.linspace(0, hparams.sampling_rate/2, hparams.filter_length//2+3)[1:-1])
#     spect = torch.from_numpy(spect).float()
#     return griffin_lim(spect, S, n_iters=n_iters, verbose=True)
def griffin_lim_synth(spect, H, n_iters=30):
    T = TacotronSTFT(
        sampling_rate=H.sampling_rate, filter_length=H.filter_length,
        hop_length=H.hop_length, win_length=H.win_length,
        n_spect_channels=H.n_spect_channels, mel_fmin=H.mel_fmin, mel_fmax=H.mel_fmax)
    spect = torch.from_numpy(spect).float()
    spect = T.mel_inv(spect)
    return griffin_lim(spect, T.stft_fn, n_iters=n_iters, verbose=True)


if __name__=='__main__':
    fire.Fire(main)
