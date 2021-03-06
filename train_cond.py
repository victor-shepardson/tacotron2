# GPU train:
# python train_cond.py -o ./checkpoints -l ./logs --n_gpus 1 --hparams "training_files='filelists/mcv_train_filelist.txt',validation_files='filelists/mcv_val_filelist.txt',batch_size=48,iters_per_checkpoint=300,load_spect_from_disk=True,n_speakers=77,speaker_embedding_dim=32,n_languages=8,language_embedding_dim=32,text_cleaners=['multi_cleaners'],symbols_embedding_dim=256,encoder_n_convolutions=4" -c checkpoints/checkpoint_19800

# CPU test:
# python train_cond.py -o ./checkpoints -l ./logs --n_gpus 0 --hparams "training_files='filelists/mcv_train_single.txt',validation_files='filelists/mcv_val_single.txt',batch_size=1,iters_per_checkpoint=5,load_spect_from_disk=True,n_speakers=60,speaker_embedding_dim=32,n_languages=2,language_embedding_dim=32,text_cleaners=['multi_cleaners'],symbols_embedding_dim=128,encoder_n_convolutions=4"

# --warm_start -c tacotron2_statedict.pt

# tensorboard --logdir checkpoints/logs > tb.log 2>&1

import os
import time
import argparse
import math
from numpy import finfo

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from fp16_optimizer import FP16_Optimizer

from model_cond import Tacotron2
from data_utils import TextMelLoader, TextMelCollate, StratifiedSampler
from loss_function import Tacotron2Loss
from logger import Tacotron2Logger
from hparams import create_hparams
from text import sequence_to_text


def batchnorm_to_float(module):
    """Converts batch norm modules to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend, init_method=hparams.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    train_sampler = (
        DistributedSampler(trainset)
        if hparams.distributed_run else
        # stratified by language
        StratifiedSampler((item[-1] for item in trainset))
        )
    # train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
    #                           sampler=train_sampler,
    #                           batch_size=hparams.batch_size, pin_memory=False,
    #                           drop_last=True, collate_fn=collate_fn)
    train_loader = DataLoader(
        trainset, batch_size=hparams.batch_size, drop_last=True,
        num_workers=4, pin_memory=True, #shuffle=True,
        collate_fn=collate_fn, sampler=train_sampler)
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams)
    if hparams.gpu:
        model = model.cuda()
    if hparams.fp16_run:
        model = batchnorm_to_float(model.half())
        model.decoder.attention_layer.score_mask_value = float(finfo('float16').min)

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    state_dict = {
        k:v for k,v in state_dict.items() if
        'decoder.attention_rnn' not in k and 'decoder.decoder_rnn' not in k
        and 'encoder.convolutions.0' not in k and 'embedding' not in k
        and 'attention' not in k and 'freq' not in k
        }
    try:
        model.load_state_dict(state_dict, False)
    except RuntimeError as err:
        print(model)
        raise err
    # checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    # model.load_state_dict(checkpoint_dict['state_dict'])
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'], False)
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, criterion, valset, iteration, batch_size, n_gpus,
             collate_fn, logger, distributed_run, rank):
    """Handles all the validation scoring and printing"""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        # val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
        #                         shuffle=False, batch_size=batch_size,
        #                         pin_memory=False, collate_fn=collate_fn)
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=4,
                                shuffle=True, batch_size=batch_size,
                                pin_memory=True, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)

            y_pred = model(x)
            loss, mel_loss, gate_loss = criterion(y_pred, y, return_parts=True)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        logger.log_validation(
            reduced_val_loss, model, y, y_pred, iteration,
            loss_parts=(mel_loss, gate_loss),
            texts=[sequence_to_text(t) for t in x[0].unbind(0)])


def train(output_directory, log_directory, checkpoint_path, warm_start, n_gpus,
          rank, group_name, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    if hparams.fp16_run:
        optimizer = FP16_Optimizer(
            optimizer, dynamic_loss_scale=hparams.dynamic_loss_scaling)

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(checkpoint_path, model)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    n_steps = 10000
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, 1e-3**(1/n_steps))

    model.train()
    # ================ MAIN TRAINING LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            scheduler.step()
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)

            y_pred = model(x)

            loss, mel_loss, gate_loss = criterion(y_pred, y, return_parts=True)
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            if hparams.fp16_run:
                optimizer.backward(loss)
                grad_norm = optimizer.clip_fp32_grads(hparams.grad_clip_thresh)
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            overflow = optimizer.overflow if hparams.fp16_run else False

            if not overflow and not math.isnan(reduced_loss) and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration,
                    loss_parts=(mel_loss, gate_loss))

            if not overflow and (iteration % hparams.iters_per_checkpoint == 0):
                validate(model, criterion, valset, iteration,
                         hparams.batch_size, n_gpus, collate_fn, logger,
                         hparams.distributed_run, rank)
                if rank == 0:
                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration))
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str,
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load the model only (warm start)')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')

    args = parser.parse_args()
    hparams = create_hparams(args.hparams)
    hparams.gpu = args.n_gpus > 0

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
