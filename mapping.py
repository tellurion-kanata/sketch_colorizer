import itertools

import torch
import torch.nn as nn
import torch.optim as optim

from models import *


class Mapper(BaseModel):
    eps = 1e-7

    def __init__(self, opt):
        super(Mapper, self).__init__(opt)
        self.initialize()

    def initialize(self):
        # Reference encoder and generator are fixed.
        self.requires_pwm = self.opt.attn_type != 'add'

        self.guide_cnn, ref_channels = define_R(
            model           = self.opt.netR,
            return_mode     = 'mapping',
            gpus            = self.opt.gpus,
            pretrained      = self.opt.pretrained_R
        )

        self.G = define_G(
            ref_channels    = ref_channels,
            ngf             = self.opt.ngf,
            attn_type       = self.opt.attn_type,
            upsample_type   = self.opt.upsample_type,
            attn_upsample   = not self.opt.no_atup,
            gpus            = self.opt.gpus
        ).eval()
        self.set_requires_grad([self.guide_cnn, self.G], False)

        self.M = define_M(
            input_ch        = 6000,
            output_ch       = ref_channels,
            n_layers        = self.opt.nlm,
            gpus            = self.opt.gpus
        )

        if self.requires_pwm:
            self.M_r = parallel(nn.Linear(ref_channels, ref_channels), gpus=self.opt.gpus)

        if not self.opt.eval:
            self.D = define_D(
                input_ch    = 4,
                n_layers    = self.opt.nld,
                ndf         = self.opt.ndf,
                spec_norm   = self.opt.use_spec,
                gpus        = self.opt.gpus
            ).eval()
            self.set_requires_grad(self.D, False)

            if self.requires_pwm:
                optim_t = itertools.chain(self.M.parameters(), self.M_r.parameters())
            else:
                optim_t = self.M.parameters()

            if self.opt.optimizer == 'sgd':
                self.optimizer_M = optim.SGD(
                    params      = optim_t,
                    lr          = self.lr,
                    momentum    = self.opt.momentum
                )
            else:
                self.optimizer_M = optim.Adam(
                    params      = optim_t,
                    lr          = self.lr,
                    betas       = self.opt.betas
                )

            self.criterion_GAN = loss.GANLoss(self.opt.gan_mode).to(self.device)
            self.criterion_L1 = nn.L1Loss().to(self.device)
            self.optimizers = [self.optimizer_M]

        self.load(models={'G': self.G}, path=self.opt.pretrained_GD)
        if self.requires_pwm:
            self.models = {'M': self.M, 'Mr': self.M_r}
        else:
            self.models = {'M': self.M}
        self.setup()


    def read_input(self, input):
        self.x = input['input'].to(self.device)
        self.x_idx = input['index']

        if self.dataset_mode == 'mapping':
            self.pr = input['pref'].to(self.device)
            self.cr = input['cref'].to(self.device)
        else:
            self.r = input['ref'].to(self.device)
            self.y = input['real'].to(self.device)


    def modify_tags(self, x):
        nx = x.clone()
        eye_list = [10, 20, 33, 43, 51, 66, 151, 195, 279, 358, 1334, 2236]
        hair_list = [11, 12, 16, 49, 63, 67, 68, 84, 92, 104, 179]
        shirt_list = [98, 453, 722, 801, 1093, 1199, 1461, 1527, 2113, 2126, 2210, 5581]
        skirt_list = [182, 269, 416, 606, 640, 948, 988, 1068, 1206, 1787, 1879, 4056]
        dress_list = [210, 230, 408, 520, 702, 812, 1097, 1785, 1993, 2162, 2978, 4463]
        sky_list = [257, 855, 3560, 4006, 4398, 4615]
        theme_list = [918, 1822, 1829, 1948, 1954, 2391, 3247, 3341, 3349, 4046, 4435, 5790]
        background = [30, 206, 290, 445, 505, 550, 754, 810, 898, 984, 1028, 1049, 1284, 1332, 2317, 2356, 2743]
        cloth = [182, 269, 416, 606, 640, 948, 988, 1068, 1206, 1787, 1879, 4056,
                 98, 453, 722, 801, 1093, 1199, 1461, 1527, 2113, 2126, 2210, 5581,
                 210, 230, 408, 520, 702, 812, 1097, 1785, 1993, 2162, 2978, 4463]
        glove_list = [142, 144, 907, 959, 1574, 1805, 2302, 2591, 2868, 2958, 3878]
        neckwear_list = [229, 468, 642, 755, 1365, 1388, 1402, 1759, 2142, 2191, 3365, 3519]
        skin_list = [158, 234, 624, 1107, 1279, 1656, 2024, 2128, 2355, 2469, 2994, 3094, 3471, 5015]

        target = hair_list + shirt_list + eye_list + skirt_list
        for idx in target:
            nx[:, idx] = 0.
        nx[:, 20] = 2.
        nx[:, 49] = 2.
        nx[:, 801] = 2.
        nx[:, 269] = 2.
        return nx


    def compute_mask(self, x):
        b, c, h, w = x.shape

        xm = x.mean(dim=[2, 3], keepdims=True)
        ones = torch.ones_like(x, device=x.device)
        x_in = torch.where(xm > 0., x/xm, ones)
        x_in = x_in.view(b, c, h * w).permute(0, 2, 1)

        m = self.M_r(x_in).permute(0, 2, 1).view(b, c, h, w)
        return m


    def forward(self):
        if self.dataset_mode == 'mapping':
            pcls, self.pfr = self.guide_cnn(self.pr)
            ccls, self.cfr = self.guide_cnn(self.cr)

            m = self.compute_mask(self.pfr) if self.requires_pwm else 1.
            self.fpm = self.M(pcls)
            self.fcm = self.M(ccls)

            bfr = self.pfr + m * (self.fcm - self.fpm)
            self.py_real, self.pfy_real = self.G(self.x, self.cfr, hook=True)
            self.py_fake, self.pfy_fake = self.G(self.x, bfr, hook=True)

        else:
            cls, fr = self.guide_cnn(self.r)
            ncls = self.modify_tags(cls)
            if torch.abs(ncls - cls).sum() > self.eps:
                m = self.compute_mask(fr) if self.requires_pwm else 1.
                fr = fr + m * (self.M(ncls) - self.M(cls))
            self.py, _ = self.G(self.x, fr)


    def backward_M(self):
        self.loss_L1 = self.criterion_L1(self.py_real, self.py_fake)
        self.loss_ft = self.criterion_L1(self.pfy_real, self.pfy_fake)
        self.loss_fr = self.criterion_L1(self.cfr.mean(dim=[2, 3], keepdims=True),
                                         self.pfr.mean(dim=[2, 3], keepdims=True) + self.fcm - self.fpm) * self.requires_pwm

        self.loss_M = self.loss_L1 + self.loss_ft + self.loss_fr
        self.loss_M.backward()


    def backward(self):
        # update mapping network
        self.optimizer_M.zero_grad()
        self.backward_M()
        self.optimizer_M.step()


    def set_evaluation_outputs(self):
        self.outputs = {
            'real': self.cr,
            'fake': self.py_fake
        }
        self.clean_img()


    def set_state_dict(self):
        self.state_dict = {
            'loss':         self.loss_M.detach().cpu().numpy(),
            'loss_L1':      self.loss_L1.detach().cpu().numpy(),
            'loss_ft':      self.loss_ft.detach().cpu().numpy(),
        }

        if self.requires_pwm:
            self.state_dict['loss_fr'] = self.loss_fr

    def log_loss_per_iter(self, bar):
        bar.set_postfix({'total_loss': self.loss_M.detach().cpu().numpy(),
                         'loss_ft': self.loss_ft.detach().cpu().numpy()})


    def get_samples(self):
        self.read_input(self.fixed_samples)
        self.forward()
        x = self.x.repeat([1, 3, 1, 1])
        self.samples = torch.cat([x, self.cr, self.py_real, self.py_fake], dim=3).cpu()


    def clean_img(self):
        del self.x, self.pr, self.cr, self.x_idx
        del self.py_real, self.py_fake, self.pfy_real, self.pfy_fake
        del self.cfr, self.pfr, self.fpm, self.fcm


    def clean_loss(self):
        del self.loss_M, self.loss_ft, self.loss_fr, self.loss_L1