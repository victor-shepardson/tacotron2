from math import sqrt
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from fp16_optimizer import fp32_to_fp16, fp16_to_fp32

def DiagonalNormal(*args):
    return D.Independent(D.Normal(*args), 1)

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_spect_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


# class Postnet(nn.Module):
#     """Postnet
#         - Five 1-d convolution with 512 channels and kernel size 5
#     """
#
#     def __init__(self, hparams):
#         super(Postnet, self).__init__()
#         self.convolutions = nn.ModuleList()
#
#         self.convolutions.append(
#             nn.Sequential(
#                 ConvNorm(hparams.n_spect_channels, hparams.postnet_embedding_dim,
#                          kernel_size=hparams.postnet_kernel_size, stride=1,
#                          padding=int((hparams.postnet_kernel_size - 1) / 2),
#                          dilation=1, w_init_gain='tanh'),
#                 nn.BatchNorm1d(hparams.postnet_embedding_dim))
#         )
#
#         for i in range(1, hparams.postnet_n_convolutions - 1):
#             self.convolutions.append(
#                 nn.Sequential(
#                     ConvNorm(hparams.postnet_embedding_dim,
#                              hparams.postnet_embedding_dim,
#                              kernel_size=hparams.postnet_kernel_size, stride=1,
#                              padding=int((hparams.postnet_kernel_size - 1) / 2),
#                              dilation=1, w_init_gain='tanh'),
#                     nn.BatchNorm1d(hparams.postnet_embedding_dim))
#             )
#
#         self.convolutions.append(
#             nn.Sequential(
#                 ConvNorm(hparams.postnet_embedding_dim, hparams.n_spect_channels,
#                          kernel_size=hparams.postnet_kernel_size, stride=1,
#                          padding=int((hparams.postnet_kernel_size - 1) / 2),
#                          dilation=1, w_init_gain='linear'),
#                 nn.BatchNorm1d(hparams.n_spect_channels))
#             )
#
#     def forward(self, x):
#         for i in range(len(self.convolutions) - 1):
#             x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
#         x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
#
#         return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional GRU
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        for i in range(hparams.encoder_n_convolutions):
            in_size = hparams.encoder_embedding_dim if i else hparams.symbols_embedding_dim
            conv_layer = nn.Sequential(
                ConvNorm(in_size,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.rnn = nn.GRU(
            hparams.encoder_embedding_dim,
            hparams.encoder_embedding_dim//2, 1,
            batch_first=True, bidirectional=True)

        self.skip_rnn = hparams.skip_rnn
        if self.skip_rnn:
            s = hparams.encoder_embedding_dim
            self.skipconv = ConvNorm(s, s,
                kernel_size=1, stride=1, padding=0, dilation=1,
                w_init_gain='relu')

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x_skip = x

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        if input_lengths is not None:
            input_lengths = input_lengths.cpu().numpy()
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # self.rnn.flatten_parameters()
        outputs, _ = self.rnn(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        if self.skip_rnn:
            outputs = outputs + self.skipconv(x_skip).transpose(1, 2)

        return outputs

    def inference(self, x, lengths=None):
        return self.forward(x, lengths)

class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_spect_channels = hparams.n_spect_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout
        self.learn_sigma_x = hparams.learn_sigma_x
        self.min_sigma_x = hparams.min_sigma_x

        if self.learn_sigma_x:
            self.out_logsigma = nn.Parameter(torch.zeros(1, self.n_spect_channels))

        self.prenet = Prenet(
            hparams.n_spect_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.GRUCell(
            hparams.prenet_dim + self.encoder_embedding_dim + hparams.latent_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.GRUCell(
            hparams.attention_rnn_dim + self.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_spect_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_spect_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, latents, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask
        self.latents = latents

    def get_latents(self, t):
        """if latents is a tuple, linear interpolation by t.
        else if latents has a time dimension, index by t."""
        if isinstance(self.latents, tuple):
            return self.latents[1]*t + self.latents[0]*(1-t)
        elif self.latents.dim() > 2:
            return self.latents[:, min(t, self.latents.shape[1]-1)]
        return self.latents

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_spect_channels, T_out) -> (B, T_out, n_spect_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_spect_channels) -> (T_out, B, n_spect_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        if isinstance(mel_outputs[0], tuple):
            # list[tuple[tensor]] (T_out, 2, B, n_spect_channels)
            # -> tensor (B, n_spect_channels, T_out)
            mel_outputs = tuple(
                torch.stack(p).permute(1, 2, 0) for p in zip(*mel_outputs))
        else:
            # list[tensor] (T_out, B, n_spect_channels)
            # -> tensor (B, n_spect_channels, T_out)
            mel_outputs = torch.stack(mel_outputs).permute(1, 2, 0)


        return mel_outputs, gate_outputs, alignments

    def mel_params(self, mel_outputs):
        """single frame of outputs -> mu, sigma"""
        mu = mel_outputs
        if self.learn_sigma_x:
            sigma = self.out_logsigma.exp().clamp(self.min_sigma_x)
            sigma = sigma.expand(mu.shape[0], *sigma.shape[1:])
        else:
            sigma = torch.ones_like(mel_outputs)*self.min_sigma_x

        return mu, sigma

    def decode(self, decoder_input, time_step):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        t = time_step#0.5 - 0.5*np.cos(min(1, time_step / 500)*np.pi)
        cell_input = torch.cat((
            decoder_input, self.attention_context, self.get_latents(t)), -1)
        self.attention_hidden = self.attention_rnn(
            cell_input, self.attention_hidden)
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden = self.decoder_rnn(
            decoder_input, self.decoder_hidden)
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        decoder_output = self.mel_params(decoder_output.squeeze(1))

        gate_prediction = self.gate_layer(decoder_hidden_attention_context).squeeze(1)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, latents, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        latents: latent variables to be fed to recurrent controller
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, latents, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        i = 0
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input, i)
            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [attention_weights]
            i+=1

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory, latents, use_gate=True, temperature=1,
            memory_lengths=None):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs
        latents: latent variables

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        mask = (
            None if memory_lengths is None
            else ~get_mask_from_lengths(memory_lengths))
        self.initialize_decoder_states(memory, latents, mask=mask)

        mel_outputs, gate_outputs, alignments = [], [], []
        i=0
        while True:
            decoder_input = self.prenet(decoder_input)
            (mu, sigma), gate_output, alignment = self.decode(decoder_input, i)

            mel_output = D.Normal(mu, sigma*temperature).sample()

            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if use_gate and torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output
            i += 1

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

class LatentEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.stride = hparams.latent_encoder_stride
        convparams = (
            hparams.latent_encoder_filters,
            hparams.latent_encoder_kernel,
            hparams.latent_encoder_stride,
            int((hparams.latent_encoder_kernel - 1) / 2))
        self.conv = nn.Sequential(
            nn.Conv1d(hparams.n_spect_channels, *convparams),
            nn.ReLU(),
            # nn.BatchNorm1d(hparams.latent_encoder_filters, affine=False),
            nn.Conv1d(hparams.latent_encoder_filters, *convparams),
            nn.ReLU(),
            # nn.BatchNorm1d(hparams.latent_encoder_filters, affine=False),
        )
        self.recurrence = nn.GRU(
            hparams.latent_encoder_filters, hparams.latent_encoder_rnn,
            bidirectional=True, batch_first=True)
        self.projection = nn.Linear(
            hparams.latent_encoder_rnn*2,
            hparams.latent_dim*2)

    def forward(self, spect, lengths):
        lengths = lengths//(self.stride**2)
        spect = spect[:,:,:lengths.max()*self.stride**2]
        x = self.conv(spect).permute(0,2,1)
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.recurrence(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        x = self.projection(x.sum(1)/lengths.float()[:,None])
        # print(x.shape)
        mu, sigma = x.chunk(2, dim=1)
        # sigma = sigma.exp() + np.exp(-3)
        sigma = F.softplus(sigma)# + np.exp(-3)
        # sigma = torch.sigmoid(sigma) + np.exp(-3)
        return mu, sigma

class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.min_sigma_z = hparams.min_sigma_z
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_spect_channels = hparams.n_spect_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.latent_encoder = LatentEncoder(hparams)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.mu = nn.Parameter(torch.randn(
            1, hparams.latent_components, hparams.latent_dim))
        self.sigma = nn.Parameter(np.log(hparams.init_sigma)*torch.ones(
            1, hparams.latent_components, hparams.latent_dim))
        # self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_input(self, inputs):
        inputs = fp32_to_fp16(inputs) if self.fp16_run else inputs
        return inputs

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask[:,None,:]#mask.expand(self.n_spect_channels, mask.size(0), mask.size(1))
            # mask = mask.permute(1, 0, 2)

            # mel mu, sigma
            outputs[0][0].data.masked_fill_(mask, 0.0)
            outputs[0][1].data.masked_fill_(mask, 0.0)
            # gate logit
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)

        outputs = fp16_to_fp32(outputs) if self.fp16_run else outputs
        return outputs

    def forward(self, inputs):
        inputs, input_lengths, targets, max_len, \
            output_lengths = self.parse_input(inputs)

        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        mu, sigma = self.latent_encoder(targets, output_lengths)
        sampled_latents = torch.randn_like(mu, requires_grad=True)*sigma + mu
        latents = mu, sigma

        q_z = DiagonalNormal(mu[:, None, :], sigma[:, None, :]) # batch x 1 x dim
        p_z = DiagonalNormal(self.mu, self.sigma.exp().clamp(self.min_sigma_z)) # 1 x components x dim
        Q_y = D.Categorical(logits=p_z.log_prob(sampled_latents[:, None, :])) # batch x components
        P_y = D.Categorical(torch.ones_like(Q_y.probs))

        kld_z = (D.kl_divergence(q_z, p_z)*Q_y.probs).mean(1)
        kld_y = D.kl_divergence(Q_y, P_y)
        kld_terms = kld_z, kld_y

        diagnostics = {
            'mean_ent': Q_y.entropy().mean(),
            'marginal_ent': D.Categorical(Q_y.probs.mean(0)).entropy()
        }

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, sampled_latents, memory_lengths=input_lengths)

        return self.parse_output(
            [mel_outputs, kld_terms, gate_outputs, alignments],
            output_lengths), diagnostics

    def inference(self, inputs, reference=None, latents=None,
            use_gate=True, input_lengths=None, reference_lengths=None, temperature=1, latent_temperature=1):
        assert (reference is None) != (latents is None)

        encoder_outputs = self.encode(inputs)

        if latents is None:
            mu, sigma = self.encode_reference(reference, reference_lengths)
            latents = D.Normal(mu, latent_temperature*sigma).sample()

        return self.decode(encoder_outputs, latents, input_lengths=input_lengths,
            use_gate=use_gate, temperature=temperature)

    def encode_reference(self, reference, reference_lengths=None):
        if reference_lengths is None:
            reference_lengths = reference.ne(0).all(2).sum(1)
        mu, sigma = self.latent_encoder(reference, reference_lengths)
        return mu, sigma

    def encode(self, inputs, input_lengths=None):
        inputs = self.parse_input(inputs)
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        return self.encoder.inference(embedded_inputs, input_lengths)

    def decode(self, encoder_outputs, latents,
            use_gate=True, temperature=1, input_lengths=None):
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs, latents,
            use_gate=use_gate, temperature=temperature, memory_lengths=input_lengths)

        outputs = self.parse_output(
            [mel_outputs, latents, gate_outputs, alignments])

        return outputs

    def sample_prior(self, n=None, y=None, z=None, temperature=1):
        if y is None:
            y = torch.randint(self.mu.shape[1], size=(n,))
        mu, sigma = (
            self.mu[0, y, :], self.sigma[0, y, :].exp().clamp(self.min_sigma_z))
        if z is None:
            p_z = DiagonalNormal(mu, temperature*sigma)
            return p_z.sample()
        else:
            return mu + z*sigma

    # def apply_postnet(self, spect):
    #     return spect + self.postnet(spect)
