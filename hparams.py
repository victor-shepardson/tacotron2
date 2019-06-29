# import tensorflow as tf
# from tensorflow.contrib.training import HParams
from text import symbols

# eliminate tf dependency. lol
class HParams(object):
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def parse(self, s):
        for line in s.split(','):
            k, v = line.split('=')
            setattr(self, k, eval(v))

    @property
    def n_spect_channels(self):
        if self.use_mel:
            return self.n_mel_channels
        return (self.filter_length//2 + 1) * (int(self.use_complex) + 1)


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        ################################
        # Experiment Parameters        #
        ################################
        gpu=True,
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,

        ################################
        # Data Parameters             #
        ################################
        load_spect_from_disk=False,
        training_files='filelists/ljs_audio_text_train_filelist.txt',
        validation_files='filelists/ljs_audio_text_val_filelist.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        use_mel=True,
        use_complex=False,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,
        n_languages=1, # cond model only
        language_embedding_dim=0, # cond model only

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        n_speakers=1, # cond model only
        speaker_embedding_dim=0, # cond model only

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        # VAE parameters
        latent_dim=16,
        latent_encoder_filters=256,
        latent_encoder_kernel=5,
        latent_encoder_stride=1,
        latent_encoder_rnn=512,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=64,
        mask_padding=True  # set model's padded outputs to padded values
    )

    if hparams_string:
        # tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

#     if verbose:
#         tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
