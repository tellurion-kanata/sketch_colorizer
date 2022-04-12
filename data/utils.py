import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image
from scipy.interpolate import make_interp_spline, BSpline
from tensorboard.backend.event_processing import event_accumulator

def to_grayscale(data):
    grayscale = transforms.Grayscale(num_output_channels=3)(data)
    return grayscale

def ema_smooth(scalars, weight=0.9):
    last = scalars[0]
    smoothed_scalars =[]
    for point in scalars:
        smoothed_t = last * weight + (1 - weight) * point
        smoothed_scalars.append(smoothed_t)
        last = smoothed_t
    return smoothed_scalars

def get_log_loss(path, key, elen=72):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    loss_log = ea.scalars.Items(key)
    step, loss = [], []
    for t in loss_log[:elen]:
        step.append(t.step)
        loss.append(t.value)
    loss = ema_smooth(loss)
    return step, loss


def prt_feat_meanval(tensors, channel_num):
    count = 0
    for t in tensors:
        t = t.mean(dim=[2, 3]).squeeze(0).detach().cpu().numpy()
        x = np.linspace(0, channel_num, channel_num)
        plt.plot(x, t, label=str(count))
        count += 1
    plt.legend()
    plt.show()


def prt_feat_map(maps, path=None):
    for cnt, map in enumerate(maps):
        filename = path + f'ch_{cnt}.jpg'
        img = (map.clone() + 1.) * 127.5
        img = img.clamp(0, 255).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = Image.fromarray(img)

        img.save(filename)

def save_image(data, filename, grayscale=False):
    """
        image should be a torch.Tensor().cpu() [c, h, w]
        rgb value: [-1, 1] -> [0, 255]
    """

    img = (data.clone() + 1.) * 127.5

    img = img.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    if grayscale:
        img = img[:, :, 0]
    img = Image.fromarray(img)
    img.save(filename)

def format_time(second):
    s = np.int(second)
    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)