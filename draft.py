import torch
import torch.nn as nn
import torch.optim as optim

from models import *


class DraftDrawer(BaseModel):
    eps = 1e-7

    def __init__(self, opt):
        super(DraftDrawer, self).__init__(opt)
        self.initialize()

    def initialize(self):
        # Reference encoder is fixed.
        self.guide_cnn, ref_channels = define_R(
            model           = self.opt.netR,
            return_mode     = self.opt.model,
            gpus            = self.opt.gpus,
            pretrained      = self.opt.pretrained_R
        )
        self.set_requires_grad(self.guide_cnn, False)

        self.G = define_G(
            ref_channels    = ref_channels,
            ngf             = self.opt.ngf,
            attn_type       = self.opt.attn_type,
            upsample_type   = self.opt.upsample_type,
            attn_upsample   = not self.opt.no_atup,
            gpus            = self.opt.gpus
        )

        if not self.opt.eval:
            self.D = define_D(
                input_ch    = 4,
                n_layers    = self.opt.nld,
                ndf         = self.opt.ndf,
                spec_norm   = self.opt.use_spec,
                gpus        = self.opt.gpus
            )

            if self.opt.optimizer == 'sgd':
                self.optimizer_G = optim.SGD(self.G.parameters(), lr=self.lr, momentum=self.opt.momentum)
                self.optimizer_D = optim.SGD(self.D.parameters(), lr=self.lr, momentum=self.opt.momentum)
            else:
                self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=self.opt.betas)
                self.optimizer_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=self.opt.betas)

            self.criterion_GAN = loss.GANLoss(self.opt.gan_mode).to(self.device)
            self.criterion_tv = loss.TVLoss().to(self.device)
            self.criterion_L1 = nn.L1Loss().to(self.device)

            self.optimizers = [self.optimizer_G, self.optimizer_D]
            self.models = {'G': self.G, 'D': self.D}
        else:
            self.models = {'G': self.G}
        self.setup()


    def read_input(self, input):
        self.x = input['input'].to(self.device)
        self.r = input['ref'].to(self.device)
        self.y = input['real'].to(self.device)
        self.x_idx = input['index']


    def forward(self):
        fr = self.guide_cnn(self.r)
        # fr = latent_shuffle(fr, self.training)
        self.py, _ = self.G(self.x, fr)


    def backward_D(self):
        fake_AB = torch.cat((self.x, self.py), 1)               # fake image
        predict_fake = self.D(fake_AB.detach())
        self.loss_D_fake = self.criterion_GAN(predict_fake, False)

        true_AB = torch.cat((self.x, self.y), 1)                # real image
        predict_true = self.D(true_AB)
        self.loss_D_true = self.criterion_GAN(predict_true, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_true) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        fake_AB = torch.cat((self.x, self.py), 1)
        predict_fake = self.D(fake_AB)
        self.loss_GAN = self.criterion_GAN(predict_fake, True)
        self.loss_L1 = self.criterion_L1(self.py, self.y) * self.opt.lambda_L1
        self.loss_tv = self.criterion_tv(self.py) * self.opt.lambda_tv

        self.loss_G = self.loss_GAN + self.loss_L1 + self.loss_tv
        self.loss_G.backward()


    def backward(self):
        # update discriminator
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update draft generator
        self.set_requires_grad(self.D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def set_evaluation_outputs(self):
        self.outputs = {
            'real': self.y,
            'fake': self.py
        }
        self.clean_img()


    def set_state_dict(self):
        self.state_dict = {
            'loss_D':       self.loss_D.detach().cpu().numpy(),
            'loss_D_true':  self.loss_D_true.detach().cpu().numpy(),
            'loss_D_fake':  self.loss_D_fake.detach().cpu().numpy(),
            'loss_G':       self.loss_G.detach().cpu().numpy(),
            'loss_GAN':     self.loss_GAN.detach().cpu().numpy(),
            'loss_L1':      self.loss_L1.detach().cpu().numpy(),
        }


    def log_loss_per_iter(self, bar):
        bar.set_postfix({'loss_G': self.loss_G.detach().cpu().numpy(),
                         'loss_D': self.loss_D.detach().cpu().numpy(),
                         'loss_L1': self.loss_L1.detach().cpu().numpy()})


    def get_samples(self):
        self.read_input(self.fixed_samples)
        self.forward()
        x = self.x.repeat([1, 3, 1, 1])
        self.samples = torch.cat([x, self.r, self.y, self.py], dim=3).cpu()


    def clean_img(self):
        del self.x, self.y, self.r, self.x_idx, self.py


    def clean_loss(self):
        del self.loss_D, self.loss_D_fake, self.loss_D_true
        del self.loss_G, self.loss_GAN, self.loss_L1, self.loss_tv
