from math import sqrt
import torch
# from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths
from fp16_optimizer import fp32_to_fp16, fp16_to_fp32


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
        attention_weights_cat = F.pad(attention_weights_cat, (1,0))
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
    # def get_alignment_energies(self, query, processed_memory):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)

        # print(processed_query.shape, processed_attention_weights.shape, processed_memory.shape)

        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))
        # energies = self.v(torch.tanh(processed_query + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
    # def forward(self, attention_hidden_state, memory, processed_memory, mask):
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
        # alignment = self.get_alignment_energies(
            # attention_hidden_state, processed_memory)

        if mask is not None:
            alignment.data.masked_fill_(mask[:, :alignment.shape[1]], self.score_mask_value)

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


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )

        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x


# class Encoder(nn.Module):
#     """Encoder module:
#         - Three 1-d convolution banks
#         - Bidirectional LSTM
#     """
#     def __init__(self, hparams):
#         super(Encoder, self).__init__()
#
#         convolutions = []
#         for _ in range(hparams.encoder_n_convolutions):
#             conv_layer = nn.Sequential(
#                 ConvNorm(hparams.encoder_embedding_dim,
#                          hparams.encoder_embedding_dim,
#                          kernel_size=hparams.encoder_kernel_size, stride=1,
#                          padding=int((hparams.encoder_kernel_size - 1) / 2),
#                          dilation=1, w_init_gain='relu'),
#                 nn.BatchNorm1d(hparams.encoder_embedding_dim))
#             convolutions.append(conv_layer)
#         self.convolutions = nn.ModuleList(convolutions)
#
#         self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
#                             int(hparams.encoder_embedding_dim / 2), 1,
#                             batch_first=True, bidirectional=True)
#
#     def forward(self, x, input_lengths):
#         for conv in self.convolutions:
#             x = F.dropout(F.relu(conv(x)), 0.5, self.training)
#
#         x = x.transpose(1, 2)
#
#         # pytorch tensor are not reversible, hence the conversion
#         input_lengths = input_lengths.cpu().numpy()
#         x = nn.utils.rnn.pack_padded_sequence(
#             x, input_lengths, batch_first=True)
#
#         self.lstm.flatten_parameters()
#         outputs, _ = self.lstm(x)
#
#         outputs, _ = nn.utils.rnn.pad_packed_sequence(
#             outputs, batch_first=True)
#
#         return outputs
#
#     def inference(self, x):
#         for conv in self.convolutions:
#             x = F.dropout(F.relu(conv(x)), 0.5, self.training)
#
#         x = x.transpose(1, 2)
#
#         self.lstm.flatten_parameters()
#         outputs, _ = self.lstm(x)
#
#         return outputs


class Decoder(nn.Module):
    """Modifed Tacotron2 decoder for unconditional generation.

    Attention is now autoregressive."""
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [hparams.prenet_dim, hparams.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention(
            hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

        aa_bottleneck = 2*hparams.encoder_embedding_dim
        self.autoattentive_layer = nn.Sequential(
            LinearNorm(
                hparams.encoder_embedding_dim, aa_bottleneck,
                bias=True, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(
                aa_bottleneck, aa_bottleneck,
                bias=True, w_init_gain='relu'),
            nn.ReLU(),
            LinearNorm(
                aa_bottleneck, hparams.encoder_embedding_dim,
                bias=True, w_init_gain='relu')
        )

    # def get_go_frame(self, memory):
    #     """ Gets all zeros frames to use as first decoder input
    #     PARAMS
    #     ------
    #     memory: decoder outputs
    #
    #     RETURNS
    #     -------
    #     decoder_input: all zeros frames
    #     """
    #     B = memory.size(0)
    #     decoder_input = Variable(memory.data.new(
    #         B, self.n_mel_channels * self.n_frames_per_step).zero_())
    #     return decoder_input

    def device(self):
        return self.prenet.layers[0].linear_layer.weight.device

    # def initialize_decoder_states(self, memory, mask):
    def initialize_decoder_states(self, B, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        # B = memory.size(0)
        # MAX_TIME = memory.size(1)

        # self.attention_hidden = Variable(memory.data.new(
        #     B, self.attention_rnn_dim).zero_())
        # self.attention_cell = Variable(memory.data.new(
        #     B, self.attention_rnn_dim).zero_())
        #
        # self.decoder_hidden = Variable(memory.data.new(
        #     B, self.decoder_rnn_dim).zero_())
        # self.decoder_cell = Variable(memory.data.new(
        #     B, self.decoder_rnn_dim).zero_())

        self.attention_hidden = torch.zeros(
            B, self.attention_rnn_dim, device=self.device(), requires_grad=True)
        self.attention_cell = torch.zeros(
            B, self.attention_rnn_dim, device=self.device(), requires_grad=True)

        self.decoder_hidden = torch.zeros(
            B, self.decoder_rnn_dim, device=self.device(), requires_grad=True)
        self.decoder_cell = torch.zeros(
            B, self.decoder_rnn_dim, device=self.device(), requires_grad=True)

        # mod: remove cumulative attention weighting scheme
        # self.attention_weights = Variable(memory.data.new(
        #     B, MAX_TIME).zero_())
        # self.attention_weights_cum = Variable(memory.data.new(
        #     B, MAX_TIME).zero_())
        # self.attention_context = Variable(memory.data.new(
        #     B, self.encoder_embedding_dim).zero_())
        self.attention_weights = torch.zeros(
            B, 0, device=self.device(), requires_grad=True)
        self.attention_weights_cum = torch.zeros(
            B, 0, device=self.device(), requires_grad=True)
        self.attention_context = torch.zeros(
            B, self.encoder_embedding_dim,
            device=self.device(), requires_grad=True)

        # mod: initialize memory/processed_memory with zeros
        # self.memory = memory
        self.memory = torch.zeros(
            B, 1, self.encoder_embedding_dim,
            device=self.device(), requires_grad=True)
        self.processed_memory = self.attention_layer.memory_layer(self.memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
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
        gate_outputs: gate output energies
        alignments:
        """
        # print(len(alignments), alignments[0])
        # print(len(gate_outputs), gate_outputs[0])
        if gate_outputs[0].dim()==0:
            gate_outputs = [g.unsqueeze(0) for g in gate_outputs]

        max_t = len(alignments)
        alignments = [F.pad(a, (0, max_t-a.size(1))) for a in alignments]
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        # gate_outputs = torch.stack(gate_outputs).view(1, -1)#.transpose(0, 1)

        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
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
        #bypass teacher forcing
        if hasattr(self, 'prev_output') and torch.rand() > 0.5:
            decoder_input = self.prenet(self.prev_output)

        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)
        # self.attention_context, self.attention_weights = self.attention_layer(
        #     self.attention_hidden, self.memory, self.processed_memory, self.mask)

        self.attention_context = (
            self.attention_context
            + self.autoattentive_layer(self.attention_context)
            )

        self.attention_weights_cum = self.attention_weights + F.pad(
            self.attention_weights_cum, (0, 1)
        )

        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(
            self.decoder_cell, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)

        self.prev_output = decoder_output.detach()

        return (decoder_output, gate_prediction, self.attention_weights)

    def update_memory(self):
        # processed_context = (
        #     self.attention_context
        #     + self.autoattentive_layer(self.attention_context)
        #     )
        # self.memory = torch.cat((
        #     self.memory, processed_context.unsqueeze(1)), dim=1)
        # self.processed_memory = torch.cat((
        #     self.processed_memory,
        #     self.attention_layer.memory_layer(processed_context).unsqueeze(1)
        #     ), dim=1)
        self.memory = torch.cat((
            self.memory, self.attention_context.unsqueeze(1)), dim=1)
        self.processed_memory = torch.cat((
            self.processed_memory,
            self.attention_layer.memory_layer(self.attention_context).unsqueeze(1)
            ), dim=1)

    # def forward(self, memory, decoder_inputs, memory_lengths):
    def forward(self, decoder_inputs, decoder_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        # decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_input = torch.zeros(
            1, *decoder_inputs.shape[:2],
            device=self.device(), requires_grad=True)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        # self.initialize_decoder_states(
        #     memory, mask=~get_mask_from_lengths(memory_lengths))
        # self.initialize_decoder_states(
        #     decoder_inputs.size(1), mask=None)
        self.initialize_decoder_states(
            decoder_inputs.size(1),
            mask=~get_mask_from_lengths(decoder_lengths, self.device()))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = (
                self.decode(decoder_input))
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
            self.update_memory()

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    # def inference(self, memory, use_gate=True):
    def inference(self, B, use_gate=True):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        # decoder_input = self.get_go_frame(memory)
        decoder_input = torch.zeros(
            B, self.n_mel_channels, device=self.device(), requires_grad=True)

        # self.initialize_decoder_states(memory, mask=None)
        self.initialize_decoder_states(B, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if use_gate and torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output
            self.update_memory()


        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        # self.embedding = nn.Embedding(
        #     hparams.n_symbols, hparams.symbols_embedding_dim)
        # std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        # val = sqrt(3.0) * std  # uniform bounds for std
        # self.embedding.weight.data.uniform_(-val, val)
        # self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def device(self):
        return self.decoder.device()

    def parse_batch(self, batch):
        # text_padded, input_lengths, mel_padded, gate_padded, \
        #     output_lengths = batch
        # text_padded = to_gpu(text_padded).long()
        # input_lengths = to_gpu(input_lengths).long()
        # max_len = torch.max(input_lengths.data).item()
        _, _, mel_padded, gate_padded, output_lengths = batch
        mel_padded = mel_padded.to(self.device(),
            non_blocking=True, dtype=torch.float)
        gate_padded = gate_padded.to(self.device(),
            non_blocking=True, dtype=torch.float)
        output_lengths = output_lengths.to(self.device(),
            non_blocking=True, dtype=torch.long)

        # return (
        #     (text_padded, input_lengths, mel_padded, max_len, output_lengths),
        #     (mel_padded, gate_padded))
        return (
            (mel_padded, output_lengths),
            (mel_padded, gate_padded))

    def parse_input(self, inputs):
        inputs = fp32_to_fp16(inputs) if self.fp16_run else inputs
        return inputs

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            # mask = ~get_mask_from_lengths(output_lengths)
            mask = ~get_mask_from_lengths(output_lengths, self.device())
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        outputs = fp16_to_fp32(outputs) if self.fp16_run else outputs
        return outputs

    def forward(self, inputs):
        # inputs, input_lengths, targets, max_len, \
        #     output_lengths = self.parse_input(inputs)
        # input_lengths, output_lengths = input_lengths.data, output_lengths.data

        targets, output_lengths = self.parse_input(inputs)
        output_lengths = output_lengths.data

        # embedded_inputs = self.embedding(inputs).transpose(1, 2)
        # encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        # mel_outputs, gate_outputs, alignments = self.decoder(
        #     encoder_outputs, targets, memory_lengths=input_lengths)
        # mel_outputs, gate_outputs, alignments = self.decoder(targets)
        mel_outputs, gate_outputs, alignments = self.decoder(targets, output_lengths)

        mel_outputs_postnet = self.apply_postnet(mel_outputs)

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    # def inference(self, inputs, use_gate=True):
        # encoder_outputs = self.encode(inputs)
        # return self.decode(encoder_outputs, use_gate=use_gate)

    # def encode(self, inputs):
    #     inputs = self.parse_input(inputs)
    #     embedded_inputs = self.embedding(inputs).transpose(1, 2)
    #     return self.encoder.inference(embedded_inputs)

    # def decode(self, encoder_outputs, use_gate=True):
    #     mel_outputs, gate_outputs, alignments = self.decoder.inference(
    #         encoder_outputs, use_gate=use_gate)
    #
    #     mel_outputs_postnet = self.apply_postnet(mel_outputs)
    #
    #     outputs = self.parse_output(
    #         [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
    #
    #     return outputs

    def inference(self, B, use_gate=True):
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            B, use_gate=use_gate)

        mel_outputs_postnet = self.apply_postnet(mel_outputs)

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs

    def apply_postnet(self, spect):
        return spect + self.postnet(spect)
