import sys, os
import numpy as np
import itertools as it

from fire import Fire
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import TextMelLoader, TextMelCollate
from hparams import create_hparams
from model_cond import Tacotron2

def main(tacotron_path, tacotron_filelist, gpu=False, batch_size=48):
    hparams = create_hparams()
    hparams.max_decoder_steps = 1000
    hparams.n_speakers = 77#97
    hparams.speaker_embedding_dim = 32
    hparams.n_languages = 8
    hparams.language_embedding_dim = 32
    hparams.symbols_embedding_dim = 256#448
    hparams.encoder_n_convolutions = 4
    hparams.gpu=gpu
    hparams.text_cleaners = ['multi_cleaners']#['transliteration_cleaners']
    hparams.load_mel_from_disk = True
    hparams.batch_size = batch_size


    text_mel_loader = TextMelLoader(tacotron_filelist, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step, return_idxs=True)
    tacotron_loader = DataLoader(
        text_mel_loader, batch_size=hparams.batch_size, drop_last=False,
        num_workers=4, pin_memory=True, shuffle=False,
        collate_fn=collate_fn)

    load_kwargs = {} if hparams.gpu else {'map_location':'cpu'}
    T = Tacotron2(hparams)
    if hparams.gpu:
        T = T.cuda()
    state_dict = torch.load(tacotron_path, **load_kwargs)['state_dict']
    T.load_state_dict(state_dict)
    T.eval()

    lines = text_mel_loader.audiopaths_and_text
    line_batches = [
        lines[i:min(len(lines), i+batch_size)]
        for i in range(0, len(lines), batch_size)]

    with torch.no_grad():
        # for (path, text, spk, lang), (batch, idxs) in zip(tqdm(lines), tacotron_loader):
        #     assert batch[-2][0]==int(spk), batch[-1][0]==int(lang)
        #     x, y = T.parse_batch(batch)
        #     mel_pred = T(x)[0][1].cpu().numpy()
        #     synth_path = path.replace('spect', 'synth_spect', 1)
        #     synth_dir = os.path.split(synth_path)[0]
        #     if not os.path.exists(synth_dir):
        #         os.mkdir(synth_dir)
        #     np.save(synth_path, mel_pred)
        for line_batch, (batch, idxs) in zip(tqdm(line_batches), tacotron_loader):
            line_batch = [line_batch[i] for i in idxs]
            x, y = T.parse_batch(batch)
            mel_preds = T(x)[1].cpu().unbind(0)
            for i, ((path, text, spk, lang), mel_pred) in enumerate(zip(line_batch, mel_preds)):
                assert batch[-2][i]==int(spk) and batch[-1][i]==int(lang), \
                    'path/output alignment is broken'
                synth_path = path.replace('spect', 'synth_spect', 1)
                synth_dir = os.path.split(synth_path)[0]
                if not os.path.exists(synth_dir):
                    os.mkdir(synth_dir)
                np.save(synth_path, mel_pred.numpy())


if __name__=='__main__':
    Fire(main)
