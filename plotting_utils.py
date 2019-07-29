import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt
import numpy as np
import scipy.stats
from matplotlib.colors import PowerNorm, SymLogNorm, Normalize


def to_pixels(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data = np.transpose(data, (2, 0, 1))
    plt.close()
    return data

def signed_rank(x):
    r = scipy.stats.rankdata(np.abs(x), method='dense').reshape(x.shape)
    return (r-1)/(r.max()-1) * np.sign(x)

def plot_multi(
        mel, attn, gate,
        text=None, target=None,
        trim=True, delta=True):
    fig = plt.figure(figsize=(12, 6+2*int(target is not None)))

    assert trim==(text is not None)

    if trim:
        text = text[:(text > 0).sum()]
        nframes = (mel>0).any(1).sum()
        if target is not None:
            # nframes = min((target>0).any(1).sum(), nframes)
            nframes = min(max(len(target), len(mel)), max((target>0).any(1).sum(), nframes))
            target = target[:nframes]
        mel = mel[:nframes]
        attn = attn[:nframes, :len(text)]
        gate = gate[:nframes]

    if target is not None:
        ax = [
            plt.axes([0.05, 0.95, 0.9, 0.05]),
            plt.axes([0.05, 0.4, 0.9, 0.55]),
            plt.axes([0.05, 0.2, 0.9, 0.2]),
            plt.axes([0.05, 0.0, 0.9, 0.2]),
        ]
    else:
        ax = [
            plt.axes([0.05, 0.95, 0.9, 0.05]),
            plt.axes([0.05, 0.3, 0.9, 0.65]),
            plt.axes([0.05, 0.0, 0.9, 0.3]),
        ]
    cb_ax = [
        plt.axes([0.95, 0.4, 0.01, 0.55]),
        # plt.axes([0.9, 0.0, 0.01, 0.2])
    ]
    cmap = 'viridis'

    g0 = ax[0].bar(np.ones_like(gate).cumsum(), gate, width=1)
    g1 = ax[1].imshow(attn.T,
        origin='lower', aspect='auto', norm=PowerNorm(0.25, 0, 1), cmap=cmap)
    g2 = ax[2].imshow(mel.T,
        origin='lower', aspect='auto', norm=PowerNorm(1, -12, 2), cmap=cmap)
    if target is not None:
        if delta:
            g3 = ax[3].imshow(np.abs(target.T - mel.T),
                origin='lower', aspect='auto',
                norm=PowerNorm(.5, 0, 10), cmap=cmap)
        else:
            g3 = ax[3].imshow(target.T,
                origin='lower', aspect='auto',
                norm=PowerNorm(1, -12, 2), cmap=cmap)
    else:
        g3 = None

    cb = [plt.colorbar(g, cax=a) for g,a in zip([g1, g3], cb_ax)]

    for a in ax[:2]:
        a.set_xticks([])

    ax[0].set_yticks([])
    ax[0].set_ylim((0,1))
    ax[0].set_xlim((0,len(mel)))

    if text is not None:
        ax[1].set_yticks(np.arange(len(text)))
        ax[1].set_yticklabels(text)
        ax[1].tick_params(length=0, labelrotation=90, labelsize=7, pad=0.5)

    ax[2].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_ylabel('output')

    if target is not None:
        ax[3].set_yticks([])
        ax[3].tick_params(length=0, pad=1, labelsize=8)
        ax[3].set_ylabel('|target - output|' if delta else 'target')

    cb[0].set_ticks(np.linspace(0, 1, 6)**2)
    cb_ax[0].tick_params(length=0, pad=1, labelsize=8)

    # cb[1].set_ticks([])
    # cb_ax[1].tick_params(length=0, pad=1, labelsize=8)

    return fig

# def save_figure_to_numpy(fig):
#     # save it to a numpy array.
#     data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     return data


def plot_alignment_to_numpy(alignment, info=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()

    return to_pixels(fig)
    # fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    # plt.close()
    # # return data
    # return np.transpose(data, (2, 0, 1))

def plot_alignments_to_numpy(alignments, info=None):
    w, h = 4, 4
    fig, axs = plt.subplots(w, h, figsize=(12, 12))
    for ax, alignment in zip(axs.flatten(), alignments):
        im = ax.imshow(alignment, aspect='auto', origin='lower',
                       interpolation='none')
        # xlabel = 'Decoder timestep'
        # if info is not None:
            # xlabel += '\n\n' + info
        # plt.xlabel(xlabel)
        # plt.ylabel('Encoder timestep')
    # plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    # return data
    return np.transpose(data, (2, 0, 1))


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    return to_pixels(fig)
    # fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    # plt.close()
    # # return data
    # return np.transpose(data, (2, 0, 1))



def plot_gate_outputs_to_numpy(gate_targets, gate_outputs):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.scatter(range(len(gate_targets)), gate_targets, alpha=0.5,
               color='green', marker='+', s=1, label='target')
    ax.scatter(range(len(gate_outputs)), gate_outputs, alpha=0.5,
               color='red', marker='.', s=1, label='predicted')

    plt.xlabel("Frames (Green target, Red predicted)")
    plt.ylabel("Gate State")
    plt.tight_layout()

    return to_pixels(fig)
    # fig.canvas.draw()
    # data = save_figure_to_numpy(fig)
    # plt.close()
    # # return data
    # return np.transpose(data, (2, 0, 1))
