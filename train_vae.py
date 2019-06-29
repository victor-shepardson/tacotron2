# CPU test:
# python train_vae.py -o ./checkpoints -l ./logs --n_gpus 0 --hparams "training_files='filelists/ljs_train_16.txt',validation_files='filelists/ljs_val_16.txt',batch_size=4,iters_per_checkpoint=10" -c tacotron2_statedict.pt --warm_start

# GPU train:
# python train_vae.py -o ./checkpoints -l ./logs --n_gpus 1 --hparams "training_files='filelists/ljs_audio_text_train_filelist.txt',validation_files='filelists/ljs_audio_text_val_filelist.txt',batch_size=50,iters_per_checkpoint=300" -c tacotron2_statedict.pt --warm_start

# -c tacotron2_statedict.pt --warm_start

# hi res LJ
# python train.py -o ./checkpoints -l ./logs --n_gpus 0 --hparams "training_files='filelists/ljs_audio_text_train_filelist.txt',validation_files='filelists/ljs_audio_text_val_filelist.txt',batch_size=2,iters_per_checkpoint=5,n_mel_channels=240,mel_fmax=11000" --warm_start -c tacotron2_statedict.pt

# python train.py -o ./checkpoints -l ./logs --n_gpus 1 --hparams "training_files='filelists/ljs_audio_text_train_filelist.txt',validation_files='filelists/ljs_audio_text_val_filelist.txt',batch_size=50,iters_per_checkpoint=300,n_mel_channels=240,mel_fmax=11000" --warm_start -c tacotron2_statedict.pt

# complex LJ
# python train.py -o ./checkpoints -l ./logs --n_gpus 0 --hparams "training_files='filelists/ljs_audio_text_train_filelist.txt',validation_files='filelists/ljs_audio_text_val_filelist.txt',batch_size=2,iters_per_checkpoint=5,use_mel=False,use_complex=True" --warm_start -c tacotron2_statedict.pt

import os
import copy
import time
import argparse
import math
from numpy import finfo
import itertools as it

import torch
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from fp16_optimizer import FP16_Optimizer

from model_vae import Tacotron2
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2VAELoss
from logger import Tacotron2Logger
from hparams import create_hparams

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

    train_sampler = DistributedSampler(trainset) \
        if hparams.distributed_run else None

    train_loader = DataLoader(trainset, num_workers=1, shuffle=True,
                              sampler=train_sampler,
                              batch_size=hparams.batch_size, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)
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


# def warm_start_model(checkpoint_path, model):
#     assert os.path.isfile(checkpoint_path)
#     print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
#     checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
#     model.load_state_dict(checkpoint_dict['state_dict'])
#     return model
def warm_start_model(checkpoint_path, model):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    # state_dict = {
    #     k:v for k,v in state_dict.items()
    #     if 'encoder' not in k and 'location' not in k and 'embedding' not in k}
    for k,v in it.chain(model.named_parameters(), model.named_buffers()):
        if k not in state_dict:
            print(f'not loading {k}')
            continue
        old_shape = state_dict[k].shape
        if v.shape != old_shape:
            print(f'ignoring "{k}" with different shape')
            state_dict[k] = v
            # print(f'resampling "{k}" from {old_shape} to {v.shape}')
            # state_dict[k] = torch.nn.functional.interpolate(
            #     state_dict[k][None, None, ...], v.shape)[0,0]
    model.load_state_dict(state_dict, False)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
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
        val_loader = DataLoader(valset, sampler=val_sampler, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            batch = batch[:5]
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                raise NotImplementedError
            else:
                reduced_val_loss = sum(loss.values()).item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        print("Validation loss {}: {:9f}  ".format(iteration, reduced_val_loss))
        y_pred[1] = y_pred[0][0]
        logger.log_validation(reduced_val_loss, model, y, y_pred, iteration)


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

    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    criterion = Tacotron2VAELoss()

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

    model.train()
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            batch = batch[:5] #compatibility with conditional model loader
            start = time.perf_counter()
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)

            loss = criterion(y_pred, y, x)
            if hparams.distributed_run:
                raise NotImplementedError
            else:
                reduced_loss = sum(loss.values())

            if hparams.fp16_run:
                optimizer.backward(reduced_loss)
                grad_norm = optimizer.clip_fp32_grads(hparams.grad_clip_thresh)
            else:
                reduced_loss.backward()

                for k,v in model.named_parameters():
                    if v.grad is None:
                        print(k, 'has no gradient')
                    else:
                        print(k, v.grad.norm())

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

            optimizer.step()

            reduced_loss = reduced_loss.item()

            overflow = optimizer.overflow if hparams.fp16_run else False

            if not overflow and not math.isnan(reduced_loss) and rank == 0:
                duration = time.perf_counter() - start
                print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                    iteration, reduced_loss, grad_norm, duration))
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)
                logger.add_scalars('training.loss', {
                    k:v.item() for k,v in loss.items()}, iteration)

            # if not overflow and (iteration % hparams.iters_per_checkpoint == 0):
            #     validate(model, criterion, valset, iteration,
            #              hparams.batch_size, n_gpus, collate_fn, logger,
            #              hparams.distributed_run, rank)
            #     if rank == 0:
            #         checkpoint_path = os.path.join(
            #             output_directory, "checkpoint_{}".format(iteration))
            #         save_checkpoint(model, optimizer, learning_rate, iteration,
            #                         checkpoint_path)

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
    hparams.gpu = torch.cuda.is_available()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
