import shutil

import torch
import torch.nn as nn
import torchvision.utils as vutils
import data.utils as utils

from torch.utils.tensorboard import SummaryWriter
from data import CustomDataLoader
from models.loss import get_scheduler
from metric import InceptionFID
from tqdm import tqdm

import os
import time


maxm_img_per_column = 16                                        # maximum sampling images of each column during training

def load_parallel_models(models, checkpoints, device):
    print('\n**************** loading model ******************')
    for net in models.keys():
        state_dict = checkpoints[net]
        required_state_dict = models[net].state_dict()
        for key in required_state_dict.keys():
            load_key = key
            if device == torch.device('cpu'):
                load_key = 'module.' + load_key
            assert load_key in state_dict.keys(), f'failed in loading model {net}, {load_key} not in the checkpoint file.'
            required_state_dict[key] = state_dict[load_key]
        models[net].load_state_dict(required_state_dict)
        print(f'\n********** load model [{net}] successfully **************')



class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.ckpt_path = os.path.join(opt.save_path,opt.name)
        self.sample_path = os.path.join(self.ckpt_path, 'sample')
        self.test_path = os.path.join(self.ckpt_path, 'test')
        self.batch_size = opt.batch_size
        self.dataset_mode = opt.dataset_mode
        self.evaluation_mode = opt.eval

        if not opt.eval:
            self.sample_size = opt.sample_batch_size if opt.sample_batch_size else self.batch_size
            self.lr = opt.learning_rate
            self.save_freq = opt.save_freq
            self.save_freq_step = opt.save_freq_step
            self.print_freq = opt.print_freq
            self.st_epoch = 1
            self.ed_epoch = opt.niter + opt.niter_decay + 1
            self.valroot = opt.valroot

        self.device = torch.device(f'cuda:{opt.gpus[0]}') if opt.gpus else torch.device('cpu')


    def setup(self):
        torch.backends.cudnn.benchmark = True
        self.data_loader = CustomDataLoader()
        self.data_loader.initialize(self.dataset_mode, self.opt)
        self.dataset = self.data_loader.load_data()
        self.data_size = self.data_loader.get_data_size()
        self.epoch_iters = len(self.dataset)

        if not self.evaluation_mode:
            self.schedulers = [get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
            self.print_param()
            self.writer = SummaryWriter(os.path.join(self.ckpt_path, 'logs'))

        self.total_time = 0.
        current_time = time.time()
        self.pre_epoch_time = current_time
        self.pre_iter_time = current_time
        self.state_dict = {}


    def train(self, mode=True):
        for net in self.models.keys():
            self.models[net].train()


    def eval(self):
        for net in self.models.keys():
            self.models[net].eval()


    def print_param(self):
        arch_log = open(os.path.join(self.ckpt_path, 'nets_arch.txt'), 'wt')
        message = ''
        param_message = ''
        all_param = 0
        for net in self.models.keys():
            message += '{:>40}:\n'.format('Network ' + net)
            params = 0
            for name, param in self.models[net].named_parameters():
                params += param.view(-1).size()[0]
                message += f'{name}: {param.size()}\n'
            params = params / 1024 ** 2
            param_message += f'network {net} parameters: {params:.5f} M\n'
            all_param += params
        param_message += f'All training parameters: {all_param:.5f} M, size {all_param*4:.2f} MB.\n'
        arch_log.write(message + '\n')

        print(param_message, end='')
        arch_log.write(param_message)


    def save(self, epoch, not_latest=False):
        save_epoch = str(epoch) + 'ep' if not_latest else 'latest'

        save_dict = {}
        save_dict['states'] = {'epoch': epoch, 'lr': self.lr, 'iterations': self.step, 'total_time': self.total_time}
        for net in self.models.keys():
            save_dict[net] = self.models[net].state_dict()
        torch.save(save_dict, os.path.join(self.ckpt_path, f'{save_epoch}_{self.opt.model}.pth'))


    def load(self, epoch='latest', resume=False, models=None, path=None):
        models = models if models else {}
        if 'G' in models.keys():                                            # 2nd training needs to load a pre-trained GAN
            default_path = os.path.join(self.ckpt_path, 'latest_colorization.pth')
            if not os.path.exists(default_path) and path:
                shutil.copy(path, default_path)
            file_path = path if path else default_path
        else:
            models = self.models
            epoch = epoch if epoch == 'latest' else epoch + 'ep'
            file_path = os.path.join(self.ckpt_path, f'{epoch}_{self.opt.model}.pth')

        try:
            checkpoints = torch.load(file_path)
        except:
            raise FileNotFoundError(f'Checkpoint file {file_path} is not found.')
        load_parallel_models(models, checkpoints, self.device)

        if resume:
            states = checkpoints['states']
            try:
                self.lr = states['lr']
                self.st_epoch = states['epoch']
                self.step = states['iterations']
                self.total_time = states['total_time']

                if epoch != 'latest':
                    self.st_epoch += 1
            except:
                print('Training states file error.')


    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        self.lr = self.optimizers[0].param_groups[0]['lr']


    def set_requires_grad(self, nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    # output training message iter ver.
    def print_training_iter(self, epoch, idx):
        current_time = time.time()
        fmt_curtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
        iter_time = current_time - self.pre_iter_time
        self.total_time += iter_time
        self.pre_iter_time = current_time

        message = f'{fmt_curtime}'
        message += f', iter_time: {utils.format_time(iter_time)}'
        message += f', total_time: {utils.format_time(self.total_time)}'
        message += f', epoch: [{epoch}/{self.ed_epoch-1}]'
        message += f', step: [{idx+1}/{self.epoch_iters}]'
        message += f', global_step: {self.step}'
        message += f', lr: {self.lr:.5f}'

        for label in self.state_dict.keys():
            message += f', {label}: {self.state_dict[label]:.6f}'
            self.writer.add_scalar(label, self.state_dict[label], global_step=self.step)

        tqdm.write(message)
        train_log = open(os.path.join(self.ckpt_path, 'train_log.txt'), 'a')
        train_log.write(message + '\n')
        train_log.close()


    # output training message epoch ver.
    def print_training_epoch(self, epoch):
        fid_message = self.fid_evaluation() if self.valroot else ''
        current_time = time.time()
        iter_time = current_time - self.pre_iter_time
        epoch_time = current_time - self.pre_epoch_time
        self.total_time += iter_time
        self.pre_iter_time = current_time
        self.pre_epoch_time = current_time

        message = '{ '
        message += f'Epcoh {epoch} finished'
        message += f',\tglobal_step: {self.step}'
        message += f',\ttotal_time: {utils.format_time(self.total_time)}'
        message += f',\tepoch_time: {utils.format_time(epoch_time)}'
        message += f',\tcurrent_lr: {self.lr:<.5f}'
        message += f'{fid_message}'
        message += ' }\n\n'

        tqdm.write(message, end='')
        train_log = open(os.path.join(self.ckpt_path, 'train_log.txt'), 'a')
        train_log.write(message)
        train_log.close()

    @torch.no_grad()
    def output_samples(self, epoch, index):
        self.eval()
        self.get_samples()
        filename = os.path.join(self.sample_path, f'ep{epoch}-step{index:05d}.png')
        self.samples = (self.samples + 1.) / 2.
        nrow = max(1, self.sample_size // maxm_img_per_column)
        vutils.save_image(self.samples, filename, nrow=nrow)
        del self.samples
        self.train()


    def clean_cache(self):
        self.clean_img()
        self.clean_loss()


    def train_loop(self):
        def get_fixed_samples():
            return next(iter(self.dataset))

        self.step = self.opt.start_step
        if self.opt.resume:
            self.load(self.opt.load_epoch, resume=True)

        self.fixed_samples = get_fixed_samples()
        for epoch in range(self.st_epoch, self.ed_epoch):
            bar = tqdm(self.dataset, desc=f'Epoch {epoch}')
            for idx, data in enumerate(bar):
                self.read_input(data)
                self.forward()
                self.backward()

                if idx % self.save_freq_step == 0:
                    self.output_samples(epoch, idx)
                    self.save(epoch)
                if idx % self.print_freq == 0:
                    self.set_state_dict()
                    self.print_training_iter(epoch, idx)

                self.log_loss_per_iter(bar)
                self.clean_cache()
                self.step += 1

            self.save(epoch)
            if epoch % self.save_freq == 0:
                self.save(epoch, True)

            self.print_training_epoch(epoch)

            for scheduler in self.schedulers:
                scheduler.step()

    # customize output samples according to the dataset type.
    def set_dict_samples(self):
        self.dict_samples = {
            'input': self.x.cpu(),
            'ref': self.r.cpu(),
            'real': self.y.cpu(),
            'fake': self.py.cpu()
        }


    @torch.no_grad()
    def test(self, epoch='latest'):
        def mk_dirs(path):
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                shutil.rmtree(path)
                os.makedirs(path)

        size_flag = '_resize' if self.opt.resize else ''
        sroot = f'{os.path.basename(self.opt.dataroot)}_{epoch}_epoch{size_flag}'

        spaths = {}
        skeys = {'fake', 'real', 'ref', 'input'} if self.opt.save_input else {'fake'}
        test_path = os.path.join(self.test_path, sroot)
        for key in skeys:
            filename = os.path.join(test_path, key)
            spaths[key] = filename
            mk_dirs(filename)

        self.load(epoch)
        self.eval()
        for data in tqdm(self.dataset, desc='Generating images'):
            self.read_input(data)
            self.forward()
            self.set_dict_samples()

            for key in spaths:
                for idx in range(len(self.x)):
                    filename = self.x_idx[idx] + '.png'
                    utils.save_image(
                        data        = self.dict_samples[key][idx],
                        filename    = os.path.join(spaths[key], filename),
                        grayscale   = key == 'input',
                    )
            del self.dict_samples


    @torch.no_grad()
    def fid_evaluation(self):
        self.eval()

        if self.evaluation_mode:
            eval_set = self.dataset
            data_size = self.data_size
        else:
            eval_loader = CustomDataLoader()
            eval_loader.initialize(self.dataset_mode, self.opt, self.valroot)
            eval_set = eval_loader.load_data()
            data_size = eval_set.get_data_size()

        net = InceptionFID(data_size=data_size, device=self.device)

        print(f'Evaluation data size {data_size}, begin to evaluate training via FID distance...')
        for data in tqdm(eval_set):
            self.read_input(data)
            self.forward()
            self.set_evaluation_outputs()
            net.accumulate_statistics_of_imgs(self.outputs['real'], target='real')
            net.accumulate_statistics_of_imgs(self.outputs['fake'], target='fake')
            net.forward_idx(self.outputs['real'].shape[0])

            del self.outputs

        fid = net.fid_distance()

        if self.evaluation_mode:
            print(f'fid_score: {fid:.6f}')
        else:
            self.train()
            return f',\tfid_score: {fid:.6f}'


    def initialize(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def read_input(self, input):
        raise NotImplementedError

    def get_samples(self):
        raise NotImplementedError

    def set_state_dict(self):
        raise NotImplementedError

    def log_loss_per_iter(self, bar: tqdm):
        raise NotImplementedError

    def clean_img(self):
        # clean inputs and outputs
        raise NotImplementedError

    def clean_loss(self):
        # clean loss functions
        raise NotImplementedError

    def set_evaluation_outputs(self):
        raise NotImplementedError
