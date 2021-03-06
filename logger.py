import sys, os, subprocess
import random
import numpy as np
import torch
from tensorboardX import SummaryWriter
# from plotting_utils import plot_alignments_to_numpy, plot_alignment_to_numpy, plot_spectrogram_to_numpy
# from plotting_utils import plot_gate_outputs_to_numpy
from plotting_utils import to_pixels, plot_multi

class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def add_loss(self, tag, loss_parts, iteration):
        if loss_parts is not None:
            self.add_scalars(tag, {
                'mel': loss_parts[0],
                'gate': loss_parts[1]
            }, iteration)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                iteration, loss_parts=None):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)
            self.add_loss("training.loss.components", loss_parts, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration,
            loss_parts=None, texts=None):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        self.add_loss("validation.loss.components", loss_parts, iteration)

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

    def log_multi(self, tag, y, y_pred, iteration):
        mel_outputs, _, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        if isinstance(mel_outputs, tuple): mel_outputs = mel_outputs[0]
        # plot alignment, mel target and predicted, gate target and predicted
        for i in range(min(3, len(mel_outputs))):
            self.add_image(f'{tag}/{i}', to_pixels(plot_multi(
                mel_outputs[i].cpu().numpy().T,
                alignments[i].cpu().numpy(),
                gate_outputs[i].cpu().numpy(),
                target=mel_targets[i].cpu().numpy().T,
                trim=False,
                delta=False
            )), iteration)
        # idx = 0#random.randint(0, alignments.size(0) - 1)
        # self.add_image(
        #     "log10(alignment+1e-3)",
        #     plot_alignment_to_numpy(
        #         (alignments[idx]+1e-3).log10().data.cpu().numpy().T),
        #     iteration)
        # self.add_image(
        #     "log10(alignment+1e-3)",
        #     plot_alignments_to_numpy(
        #         (alignments+1e-3).log10().data.cpu().numpy().transpose(0,2,1)),
        #     iteration)
        # self.add_image(
        #     "mel_target",
        #     plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
        #     iteration)
        # self.add_image(
        #     "mel_predicted",
        #     plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
        #     iteration)
        # self.add_image(
        #     "mel_delta",
        #     plot_spectrogram_to_numpy(
        #         (mel_targets[idx]-mel_outputs[idx]).data.cpu().numpy()),
        #     iteration)
        # self.add_image(
        #     "gate",
        #     plot_gate_outputs_to_numpy(
        #         gate_targets[idx].data.cpu().numpy(),
        #         torch.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
        #     iteration)
        # if texts:
            # self.add_text('texts', '\n'.join(texts), iteration)

if __name__=='__main__':
    print('starting tensorboard')
    subprocess.run(
        f'tensorboard --logdir {sys.argv[1]} > tensorboard.log 2>&1',
        shell=True)
