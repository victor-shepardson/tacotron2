import torch
from torch import nn
import torch.nn.functional as F

class Tacotron2Loss(nn.Module):
    def __init__(self, use_mel=True):
        super(Tacotron2Loss, self).__init__()
        self.use_mel = use_mel

    def forward(self, model_output, targets, return_parts=False, use_mel=True):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)

        if not self.use_mel:
            # quick n dirty weights for linear spectrogram
            n_bins = mel_target.shape[1]
            bin_weights = 2**(torch.linspace(
                -.1, 1, n_bins, device=mel_out.device
                ).clamp(0)*-6)+0.05
            bin_weights[0] = 0.05
            bin_weights = bin_weights[None, :, None]

            mel_loss = torch.mean((
                (mel_out - mel_target)**2
                + (mel_out_postnet - mel_target).abs()
                ) * bin_weights)
        else:
            mel_loss = F.mse_loss(mel_out, mel_target) \
                + F.mse_loss(mel_out_postnet, mel_target)

        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        if return_parts:
            return mel_loss + gate_loss, mel_loss.item(), gate_loss.item()
        return mel_loss + gate_loss
