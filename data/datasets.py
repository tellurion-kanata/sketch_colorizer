import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

import numpy.random as random
import PIL.Image as Image

from os.path import *
from glob import glob
from PIL import ImageFile

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_transform_seeds(seed_range, img_size=512, crop_size=None, rotate_p=0.2):
    seeds = [random.randint(-seed_range, seed_range),
             random.randint(-seed_range, int(seed_range*rotate_p))]
    if crop_size is None:
        return seeds
    if crop_size == img_size:
        return seeds, None
    top, left = random.randint(0, img_size - crop_size, 2)
    crops = [top, left, crop_size]

    return seeds, crops


def custom_transform(img, seeds, opt, crops=None, is_ref=True):
    range_seed, rotate_flag = seeds[:]
    if not opt.no_flip and range_seed > 0:
        img = tf.hflip(img)
    if not opt.no_rotate and is_ref and rotate_flag > 0:
        img = tf.rotate(img, range_seed, fill=255)
    if not opt.no_crop and crops is not None:
        top, left, length = crops[:]
        img = tf.crop(img, top, left, length, length)
    if not opt.no_resize:
        img = tf.resize(img, opt.load_size)
    return img


def jitter(img, seeds):
    brt, crt, sat = seeds[:]
    img = tf.adjust_brightness(img, brt)
    img = tf.adjust_contrast(img, crt)
    img = tf.adjust_saturation(img, sat)
    return img


def normalize(img, grayscale=False):
    img = transforms.ToTensor()(img)
    if grayscale:
        img = transforms.Normalize((0.5), (0.5))(img)
    else:
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    return img


class DraftDataset(data.Dataset):
    def __init__(self, opt, valroot=None):
        def exists_property(t, pname):
            return getattr(t, pname) if hasattr(t, pname) else False

        self.opt = opt
        self.eval = True if valroot else opt.eval
        dataroot = valroot if valroot else opt.dataroot

        if not exists(opt.dataroot):
            raise FileNotFoundError('Data file is not found in {}.'.format(opt.dataroot))

        self.sketch_root = join(dataroot, 'sketch')
        self.color_root = join(dataroot, 'color')
        self.reference_root = join(dataroot, 'reference')
        self.image_files = [file for ext in IMAGE_EXTENSIONS
                            for file in glob(join(self.sketch_root, '*.{}'.format(ext)))]

        if not self.eval:
            self.image_size = opt.image_size                                                    # image size in training dataset
            self.crop_size = max(self.opt.load_size, round(self.image_size * opt.crop_scale))
        else:
            self.offset = -1
            if not opt.no_ref_shuffle:
                self.data_size = len(self)
                self.offset = random.randint(1, self.data_size)
            self.resize = exists_property(opt, 'resize')
            self.save_input = exists_property(opt, 'save_input')

    def __getitem__(self, index):
        image_file = self.image_files[index]
        filename = basename(image_file)
        img_idx = splitext(filename)[0]

        color = None
        sketch = Image.open(image_file).convert('L')
        ref = Image.open(join(self.reference_root, filename)).convert('RGB')

        if not self.eval:
            color = Image.open(join(self.color_root, filename)).convert('RGB')
            # flip, crop and resize in custom transform function
            seed, crops = get_transform_seeds(1, self.image_size, self.crop_size)
            color = custom_transform(color, seed, self.opt, crops, is_ref=False)
            sketch = custom_transform(sketch, seed, self.opt, crops, is_ref=False)

            # no crop for reference image
            seed = get_transform_seeds(90)
            ref = custom_transform(ref, seed, self.opt)

            # change brightness, contrast and saturation in jitter function
            if self.opt.jitter:
                seed_j = random.random(3) * 0.2 + 0.9
                ref = jitter(ref, seed_j)
        else:
            if self.save_input or self.offset > 0:
                color = Image.open(join(self.color_root, filename)).convert('RGB')
            if self.offset > 0:
                ref_file = self.image_files[(index + self.offset) % self.data_size]
                filename = basename(ref_file)
                ref = Image.open(join(self.color_root, filename)).convert('RGB')
            elif self.offset == 0:
                color = Image.open(join(self.color_root, filename)).convert('RGB')
            if self.resize:
                sketch = transforms.Resize((self.opt.load_size, self.opt.load_size))(sketch)
            h, w = sketch.size
            ref = transforms.Resize((h, w))(ref)

        sketch = normalize(sketch, grayscale=True)
        ref =  normalize(ref)
        color = normalize(color) if color else sketch
        return {
            'input':    sketch,
            'ref':      ref,
            'real':     color,
            'index':    img_idx
        }

    def __len__(self):
        return len(self.image_files)

    def get_return_keys(self):
        return ['input', 'ref', 'real']


class MappingDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.eval = opt.eval

        if not exists(self.opt.dataroot):
            raise FileNotFoundError('Data file is not found.')
        self.sketch_root = join(self.opt.dataroot, 'sketch')
        self.color_root = join(self.opt.dataroot, 'color')
        self.image_files = [file for ext in IMAGE_EXTENSIONS
                            for file in glob(join(self.color_root, '*.{}'.format(ext)))]

        self.data_size = len(self)
        self.offset = random.randint(1, self.data_size - 1)                                     # offset used for evaluation
        if not self.eval:
            self.image_size = opt.image_size                                                    # image size in training dataset
            self.crop_size = max(self.opt.load_size, round(self.image_size * opt.crop_scale))
        else:
            try:
                self.resize = self.opt.resize
            except:
                self.resize = True


    def __getitem__(self, index):
        image_file = self.image_files[index]
        filename = basename(image_file)
        img_idx = splitext(filename)[0]
        sketch = Image.open(join(self.sketch_root, filename)).convert('L')

        if not self.eval:
            seed1, seed2 = random.randint(0, self.data_size), random.randint(0, self.data_size)
            if seed1 == seed2:
                seed2 = (seed1 + 1) % self.data_size
            ref_file1, ref_file2 = self.image_files[seed1], self.image_files[seed2]
            pref = Image.open(ref_file1).convert('RGB')
            cref = Image.open(ref_file2).convert('RGB')

            seed, crops = get_transform_seeds(1, self.image_size, self.crop_size)
            sketch = custom_transform(sketch, seed, self.opt, crops, is_ref=False)

            # no crop for reference image
            seed = get_transform_seeds(90)
            pref = custom_transform(pref, seed, self.opt)
            cref = custom_transform(cref, seed, self.opt)

            # change brightness, contrast and saturation in jitter function
            if self.opt.jitter:
                seed_j = random.random(3) * 0.2 + 0.9
                pref = jitter(pref, seed_j)
                cref = jitter(cref, seed_j)
        else:
            cref_file = self.image_files[(index+self.offset)%self.data_size]
            pref = Image.open(image_file).convert('RGB')
            cref = Image.open(cref_file).convert('RGB')

            if self.resize:
                sketch = transforms.Resize((self.opt.load_size, self.opt.load_size))(sketch)
            h, w = sketch.size
            pref = transforms.Resize((h, w))(pref)
            cref = transforms.Resize((h, w))(cref)

        sketch = normalize(sketch, grayscale=True)
        pref = normalize(pref)
        cref = normalize(cref)

        return {
            'input':    sketch,
            'pref':     pref,
            'cref':     cref,
            'index':    img_idx
        }

    def __len__(self):
        return len(self.image_files)

    def get_return_keys(self):
        return ['input', 'pref', 'cref']


class CustomDataLoader():
    def initialize(self, mode, opt, valroot=None):
        if mode == 'colorization':
            self.dataset = DraftDataset(opt, valroot)
        elif mode == 'mapping':
            self.dataset = MappingDataset(opt)
        else:
            raise NotImplementedError('DataLoader {} is not implemented.'.format(mode))

        self.dataLoader = data.DataLoader(
            dataset     = self.dataset,
            batch_size  = opt.batch_size,
            shuffle     = not opt.no_shuffle and not opt.eval,
            num_workers = opt.num_threads,
            drop_last   = len(opt.gpus) > 1
        )

    def load_data(self):
        return self

    def get_data_size(self):
        return len(self.dataset)

    def get_keys(self):
        try:
            return self.dataset.get_return_keys()
        except:
            raise NotImplementedError

    def __iter__(self):
        for data in self.dataLoader:
            yield data

    def __len__(self):
        return len(self.dataLoader)