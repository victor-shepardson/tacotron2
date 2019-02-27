# to create executable:
# > conda activate waveglow
# > source bundle.fish

# ## Tacotron 2 + Waveglow inference code

import sys
import multiprocessing

# pyinstaller fix
multiprocessing.freeze_support()

import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from train import load_model
from text import text_to_sequence
from librosa.output import write_wav

sys.path.append('waveglow/')

def main():
    # #### Setup hparams

    hparams = create_hparams()
    hparams.sampling_rate = 22050
    hparams.gpu = False

    device = torch.device('cuda') if hparams.gpu else torch.device('cpu')


    # #### Load model from checkpoint

    checkpoint_path = "tacotron2_statedict.pt"
    model = load_model(hparams)
    load_kwargs = {} if hparams.gpu else {'map_location':'cpu'}
    model.load_state_dict(torch.load(checkpoint_path, **load_kwargs)['state_dict'])
    _ = model.eval()


    # #### Load WaveGlow for mel2audio synthesis

    waveglow_path = 'waveglow_old.pt'
    waveglow = torch.load(waveglow_path, **load_kwargs)['model']
    waveglow.to(device)


    # #### Prepare text input

    # text = "Waveglow is really awesome!"
    # text = "Ppupapupplbbf. bb."
    # text = "p. p. p. hngeeeehh."
    text = ' '.join(sys.argv[1:])
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device).long()


    # #### Decode text input and plot results

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

    # #### Synthesize audio from spectrogram using WaveGlow

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    write_wav('out.wav', audio[0].data.cpu().numpy(), hparams.sampling_rate)

if __name__=='__main__':
    main()
