import os
import argparse
import torch

class Options():
    def __init__(self, eval=False):
        self.initialize(eval)
        self.modify_options()

    def initialize(self, eval=False):
        self.eval = eval

        self.parser = argparse.ArgumentParser()
        # overall options
        self.parser.add_argument('--name', required=True,
                                 help='Project name under the checkpoints file')
        self.parser.add_argument('--model', '-m', default='colorization', choices=['colorization', 'mapping', 'tag'],
                                 help='Network model type [colorization | mapping | tag]')
        self.parser.add_argument('--dataroot', '-d', type=str, required=True,
                                 help='Training dataset')
        self.parser.add_argument('--gpus', type=str, default='0',
                                 help='gpu ids:  e.g. 0 | 0,1 | 0,2 | -1 for cpu')
        self.parser.add_argument('--batch_size', '-bs', default=32, type=int,
                                 help='Number of batch size')
        self.parser.add_argument('--load_epoch', '-le', type=str, default='latest',
                                 help='Epoch to load. Default is \'latest\'')
        self.parser.add_argument('--load_size', type=int, default=384,
                                 help='Loaded size of image')
        self.parser.add_argument('--num_threads', '-nt', type=int, default=0,
                                 help='Number of threads when reading data')
        self.parser.add_argument('--save_path', '-s', type=str, default='./checkpoints',
                                 help='Trained models save path')
        self.parser.add_argument('--pretrained_R', '-pre', type=str, default='pretrained/resnet34_512px.pth',
                                 help='Pre-trained reference encoder path')
        self.parser.add_argument('--pretrained_GD', '-pg', type=str, default=None,
                                 help='Pre-trained generator and discriminator [Mapping model]')
        self.parser.add_argument('--no_shuffle', action='store_true',
                                 help='Not to shuffle data every epoch')
        self.parser.add_argument('--no_ref_shuffle', action='store_true',
                                 help='Shuffle reference image during evaluation or testing')

        # Generator settings
        self.parser.add_argument('--netR', type=str, default='resnet34', choices=['resnet34', 'resnet50', 'resnext50'],
                                 help='Type of guide network for extracting feature from reference image [ resnet34 | resnet50 | resnext50 ]')
        self.parser.add_argument('--ngf', type=int, default=64,
                                 help='Channel size base of Generator [64 | 32]')
        self.parser.add_argument('--nlm', type=int, default=4,
                                 help='Layer number of Mapping network')
        self.parser.add_argument('--attn_type', '-at', type=str, default='fc', choices=['add', 'fc'],
                                 help='Attention module type [add | fc]')
        self.parser.add_argument('--upsample_type', '-up', type=str, default='subpixel', choices=['interpolate', 'subpixel'],
                                 help='Upsampling block used in Generator [interpolate | subpixel]')
        self.parser.add_argument('--no_atup', action='store_true',
                                 help='Not to use reference-based attention upsample block')

        # Discriminator settings
        self.parser.add_argument('--ndf', type=int, default=64,
                                 help='Channel size base of Discriminator [64 | 96]')
        self.parser.add_argument('--nld', type=int, default=3,
                                 help='Layer number of Discriminator')
        self.parser.add_argument('--use_spec', action='store_true',
                                 help='Use spectral normalization in discriminator')


    def modify_options(self):
        if not self.eval:
            self.add_training_options()
        else:
            self.parser.add_argument('--resize', action='store_true',
                                     help='Resize image to opt.load_size during testing')
            self.parser.add_argument('--save_input', '-si', action='store_true',
                                     help='Save input images, testing batch size must be 1 to avoid reading error.')
        return self.parser


    def add_training_options(self):
        self.parser.add_argument('--resume', action='store_true',
                                 help='Resume training')

        # training options
        self.parser.add_argument('--sample_batch_size', '-sbs', default=None, type=int,
                                 help='Number of batch size for sampling')
        self.parser.add_argument('--niter', type=int, default=9,
                                 help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4,
                                 help='Initial learning rate')
        self.parser.add_argument('--gan_mode', type=str, default='vanilla', choices=['vanilla', 'lsgan', 'wgangp'],
                                 help='Type of GAN loss function [vanilla (default) | lsgan | wgangp]')
        self.parser.add_argument('--lr_policy', type=str, default='lambda',
                                 help='Policy of learning rate decay')
        self.parser.add_argument('--optimizer', '-opt', default='adam', choices=['sgd', 'adam'],
                                 help='Optimizer used in training [sgd | adam]')
        self.parser.add_argument('--betas', type=float, default=(0.5, 0.99),
                                 help='Betas for Adam')
        self.parser.add_argument('--momentum', default=0.9,
                                 help='Momentum used for SGD')
        self.parser.add_argument('--start_step', type=int, default=0,
                                 help='Start step')
        self.parser.add_argument('--valroot', '-vr', type=str, default=None,
                                 help='Validation dataset')

        # image pre-processing and training states output
        self.parser.add_argument('--image_size', type=int, default=512,
                                 help='Original size of training image')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='Not to flip image')
        self.parser.add_argument('--no_crop', action='store_true',
                                 help='Not to crop image')
        self.parser.add_argument('--no_rotate', action='store_true',
                                 help='Not to rotate image')
        self.parser.add_argument('--no_resize', action='store_true',
                                 help='Not to resize image during training')
        self.parser.add_argument('--jitter', action='store_true',
                                 help='Applying color adjustment during training')
        self.parser.add_argument('--crop_scale', type=float, default=0.75,
                                 help='Scale of crop operation.')
        self.parser.add_argument('--save_freq', type=int, default=1,
                                 help='Saving network states per epochs')
        self.parser.add_argument('--save_freq_step', type=int, default=5000,
                                 help='Saving latest network states per steps')
        self.parser.add_argument('--print_freq', type=int, default=1000,
                                 help='Print training states per iterations')

        # lambdas for loss functions
        self.parser.add_argument('--lambda_L1', type=float, default=100.0,
                                 help='Lambda for L1 loss function')
        self.parser.add_argument('--lambda_tv', type=float, default=1e-4,
                                 help='Lambda for total variation loss function')


    def mkdirs(self, opt):
        def makedir(paths):
            for p in paths:
                if not os.path.exists(p):
                    os.mkdir(p)

        opt.ckpt_path = os.path.join(opt.save_path, opt.name)
        opt.sample_path = os.path.join(opt.ckpt_path, 'sample')
        opt.test_path = os.path.join(opt.ckpt_path, 'test')
        makedir([opt.save_path, opt.ckpt_path, opt.sample_path, opt.test_path])

    def parse(self, opt):
        if opt.dataroot[-1] == '\\':
            opt.dataroot = opt.dataroot[:-1]
        str_ids = opt.gpus.split(',')
        opt.gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpus.append(id)
        if len(opt.gpus) > 0:
            torch.cuda.set_device(opt.gpus[0])


    def print_options(self, opt, phase):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        mode = 'at' if not self.eval else 'wt'
        file_name = os.path.join(opt.ckpt_path, '{}_opt.txt'.format(phase))
        with open(file_name, mode) as opt_file:
            opt_file.write(message)
            opt_file.write('\n')


    def get_options(self):
        opt = self.parser.parse_args()
        opt.eval = self.eval

        self.mkdirs(opt)
        self.parse(opt)

        return opt