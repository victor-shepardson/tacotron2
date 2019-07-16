import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
from utils import get_mask_from_lengths

class Tacotron2GMVAELoss(nn.Module):
    def __init__(self, use_mel=True, cycle_xform=None):
        super().__init__()

    def forward(self, hparams, model_output, targets, diagnostics, x=None, orig_out_lens=None):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, kld_terms, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)

        device = alignments.device

        attn_loss = 0
        if x is not None:
            in_lens, out_lens = x[1], x[4]
            # batch x out x in
            i = torch.arange(
                alignments.shape[1], dtype=torch.float32,
                device=device)[None, :, None]
            j = torch.arange(
                alignments.shape[2], dtype=torch.float32,
                device=device)[None, None, :]
            m = get_mask_from_lengths(out_lens, device).float()[:, :, None]
            s = (in_lens.float() / orig_out_lens.float())[:, None, None]
            sigma, margin = 30, 10
            w = 1-torch.exp(-(((j-i*s).abs()-margin).clamp(0)/sigma)**2)
            attn_loss = (w*alignments*m).sum(2).mean()

        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # ll_loss = -D.Normal(*mel_out).log_prob(mel_target)
        # mu, sigma = (t.permute(0,2,1) for t in mel_out)
        # ll_loss = -(
        #     D.Independent(D.Normal(mu, sigma), 1)
        #     .log_prob(mel_target.permute(0,2,1)))
        mu, sigma = mel_out
        mse_loss = ((mel_target - mu)*sigma).pow(2).mean()

        kld_z, kld_y = kld_terms

        r = dict(
            gate_loss = gate_loss,
            attn_loss = attn_loss,
            mse_loss = mse_loss.mean(),
            zkl_loss = kld_z.mean(),
            ykl_loss = kld_y.mean(),
        )

        if hparams.marginal_entropy_weight != 0:
            r['neg_marginal_entropy'] = hparams.marginal_entropy_weight*(
                np.log(hparams.latent_components) - diagnostics['marginal_ent'])

        # print({k:float(v) for k,v in r.items()})
        return r

class Tacotron2VAELoss(nn.Module):
    def __init__(self, use_mel=True, cycle_xform=None):
        super().__init__()

    def forward(self, model_output, targets, x=None):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, latents, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)

        device = alignments.device

        attn_loss = 0
        if x is not None:
            in_lens, out_lens = x[1], x[4]
            # batch x out x in
            i = torch.arange(
                alignments.shape[1], dtype=torch.float32,
                device=device)[None, :, None]
            j = torch.arange(
                alignments.shape[2], dtype=torch.float32,
                device=device)[None, None, :]
            m = get_mask_from_lengths(out_lens, device).float()[:, :, None]
            s = (in_lens.float() / out_lens.float())[:, None, None]
            sigma, margin = 30, 10
            w = 1-torch.exp(-(((j-i*s).abs()-margin).clamp(0)/sigma)**2)
            attn_loss = (w*alignments*m).sum(2).mean()

        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # ll_loss = -D.Normal(*mel_out).log_prob(mel_target)
        # mu, sigma = (t.permute(0,2,1) for t in mel_out)
        # ll_loss = -(
        #     D.Independent(D.Normal(mu, sigma), 1)
        #     .log_prob(mel_target.permute(0,2,1)))
        mu, sigma = mel_out
        ll_loss = ((mel_target - mu)*sigma).pow(2).mean()

        mu, sigma, latent_samples = latents
        # Q = D.Normal(mu, sigma)
        # P = D.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        # kl_loss = Q.log_prob(latent_samples) - P.log_prob(latent_samples)
        Q = D.Independent(D.Normal(mu, sigma), 1)
        P = D.Independent(D.Normal(torch.zeros_like(mu), torch.ones_like(sigma)), 1)
        kl_loss = D.kl_divergence(Q, P)

        r = dict(
            gate_loss = gate_loss,
            attn_loss = attn_loss,
            ll_loss = ll_loss.mean(),
            kl_loss = kl_loss.mean(),
        )
        # print({k:float(v) for k,v in r.items()})
        return r


class Tacotron2Loss(nn.Module):
    def __init__(self, use_mel=True, cycle_xform=None):
        super(Tacotron2Loss, self).__init__()
        self.use_mel = use_mel
        self.cycle_xform = cycle_xform

    def forward(self, model_output, targets, x=None, return_parts=False):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, alignments = model_output
        gate_out = gate_out.view(-1, 1)

        device = mel_out.device

        attn_loss = 0
        if x is not None:
            in_lens, out_lens = x[1], x[4]
            # batch x out x in
            i = torch.arange(
                alignments.shape[1], dtype=torch.float32,
                device=device)[None, :, None]
            j = torch.arange(
                alignments.shape[2], dtype=torch.float32,
                device=device)[None, None, :]
            m = get_mask_from_lengths(out_lens, device).float()[:, :, None]
            s = (in_lens.float() / out_lens.float())[:, None, None]
            sigma, margin = 30, 10
            w = 1-torch.exp(-(((j-i*s).abs()-margin).clamp(0)/sigma)**2)
            attn_loss = (w*alignments*m).sum(2).mean()
            print(attn_loss.item())

        if not self.use_mel:
            # quick n dirty weights for linear spectrogram
            n_bins = mel_target.shape[1]
            if self.cycle_xform is not None:
                n_bins = n_bins//2
            bin_weights = 2**(torch.linspace(
                -.1, 1, n_bins, device=mel_out.device
                ).clamp(0)*-6)+0.05
            bin_weights[0] = 0.05
            bin_weights = bin_weights[None, :, None]
            if self.cycle_xform is not None:
                bin_weights = torch.cat((bin_weights, bin_weights), dim=1)

            # mel_loss = torch.mean((
            #     (mel_out - mel_target)**2
            #     + (mel_out_postnet - mel_target).abs()
            #     ) * bin_weights)
            prenet_loss = torch.mean(
                (mel_out - mel_target)**2 * bin_weights)
            postnet_loss = torch.mean(
                (mel_out - mel_target).abs() * bin_weights)
            mel_loss = prenet_loss + postnet_loss
            if self.cycle_xform is not None:
                consistency_loss = F.mse_loss(
                    mel_out_postnet,
                    self.cycle_xform.reproject(mel_out_postnet))
                mel_loss = mel_loss + consistency_loss
                print(prenet_loss.item(), postnet_loss.item(), consistency_loss.item())
        else:
            mel_loss = F.mse_loss(mel_out, mel_target) \
                + F.mse_loss(mel_out_postnet, mel_target)

        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        if return_parts:
            return mel_loss + gate_loss, mel_loss.item(), gate_loss.item()
        return mel_loss + gate_loss + attn_loss
