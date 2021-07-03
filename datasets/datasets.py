import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as tf

import os
import numpy.random as random
from glob import glob
import PIL.Image as Image
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
def get_transforms(opt):
    transform_list = []
    ref_transforms_list = []
    if not opt.no_resize:
        transform_list += [transforms.Resize((opt.load_size, opt.load_size))]
    if not opt.no_flip:
        ref_transforms_list += [transforms.RandomHorizontalFlip()]
    if not opt.no_rotate:
        ref_transforms_list += [transforms.RandomRotation((0, 45))]
    if not opt.no_coljitter:
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.8)
        transform_list += [color_jitter]

    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def get_transform_seeds(load_size):
    seed = random.randint(-45, 45)
    crop_len = random.randint(load_size, 512)
    top, left = random.randint(0, 512 - crop_len, 2)
    crops = [top, left, crop_len]

    return seed, crops


def custom_transform(img, seed, crops, opt):
    top, left, length = crops[:]
    img = transforms.ToTensor()(img)
    if not opt.no_rotate:
        img = tf.rotate(img, seed)
    if not opt.no_flip and seed >= 0:
        img = tf.hflip(img)
    if not opt.no_crop:
        img = tf.crop(img, top, left, length, length)
    if not opt.no_resize:
        img = tf.resize(img, opt.load_size)
    return img


def jitter(img, seeds):
    brt, crt, sat = seeds[:]
    img = tf.adjust_brightness(img, brt)
    img = tf.adjust_contrast(img, crt)
    img = tf.adjust_saturation(img, sat)
    img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
    return img


class DraftDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        image_pattern = opt.image_pattern
        if not os.path.exists(self.opt.dataroot):
            raise FileNotFoundError('data file is not found.')
        self.sketch_root = os.path.join(self.opt.dataroot, 'sketch')
        self.color_root = os.path.join(self.opt.dataroot, 'color')
        self.reference_root = os.path.join(self.opt.dataroot, 'reference')
        self.image_files = glob(os.path.join(self.color_root, image_pattern))


    def __getitem__(self, index):
        image_file = self.image_files[index]
        color = Image.open(image_file).convert('RGB')
        sketch = Image.open(image_file.replace(self.color_root, self.sketch_root)).convert('RGB')
        ref = Image.open(image_file.replace(self.color_root, self.reference_root)).convert('RGB')

        seed, crops = get_transform_seeds(self.opt.load_size)
        color = custom_transform(color, seed, crops, self.opt)
        sketch = custom_transform(sketch, seed, crops, self.opt)

        seed, crops = get_transform_seeds(self.opt.load_size)
        ref = custom_transform(ref, seed, crops, self.opt)

        seeds = random.random(4) + 0.5
        color, ref = jitter(color, seeds), jitter(ref, seeds)

        return {'sketch': sketch,
                'reference': ref,
                'color': color,
                'index': index}

    def __len__(self):
        return len(self.image_files)


class RefineDataset():
    def __init__(self, opt):
        self.opt = opt
        if opt.no_resize:
            self.nsize = None
        else:
            self.nsize = opt.load_size

        image_pattern = opt.image_pattern
        if not os.path.exists(self.opt.dataroot):
            raise FileNotFoundError('data file is not found.')
        self.color_root = os.path.join(self.opt.dataroot, 'color')
        self.draft_root = os.path.join(self.opt.dataroot, 'spray')
        self.reference_root = os.path.join(self.opt.dataroot, 'reference')
        self.image_files = glob(os.path.join(self.color_root, image_pattern))


    def __getitem__(self, index):
        image_file = self.image_files[index]
        color = Image.open(image_file).convert('RGB')
        draft = Image.open(image_file.replace(self.color_root, self.draft_root)).convert('RGB')
        ref = Image.open(image_file.replace(self.color_root, self.reference_root)).convert('RGB')

        seed, crops = get_transform_seeds(self.opt.load_size)
        color = custom_transform(color, seed, crops, self.opt)
        draft = custom_transform(draft, seed, crops, self.opt)

        seed, crops = get_transform_seeds(self.opt.load_size)
        ref = custom_transform(ref, seed, crops, self.opt)

        seeds = random.random(3) + 0.5
        color, draft, ref = jitter(color, seeds), jitter(draft, seeds), jitter(ref, seeds)

        return {'draft': draft,
                'reference': ref,
                'color': color,
                'index': index}


    def __len__(self):
        return len(self.image_files)


class TestDataset():
    def __init__(self, opt):
        self.opt = opt
        self.opt.no_flip = True
        self.num_classes = self.opt.num_classes
        self.image_root = os.path.join(self.opt.dataroot, 'test')
        self.image_files = glob(os.path.join(self.image_root, '*/*.jpg'))
        self.incl_label = self.opt.gt_label

        if self.incl_label:
            self.tags_root = os.path.join(self.opt.dataroot, 'tags')

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert('RGB')

    def __len__(self):
        return len(self.image_files)

class CustomDataLoader():
    def initialize(self, opt):
        if not opt.eval:
            if opt.model == 'draft':
                self.dataset = DraftDataset(opt)
            elif opt.model == 'refine':
                self.dataset = RefineDataset(opt)
            else:
                raise NotImplementedError('Such model is not implemented.')
        else:
            self.dataset = RefineDataset(opt)

        self.dataLoader = data.DataLoader(
            dataset = self.dataset,
            batch_size = opt.batch_size,
            shuffle = not opt.no_shuffle and not opt.eval,
            num_workers = opt.num_threads)

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for data in self.dataLoader:
            yield data