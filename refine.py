import torch
import torch.nn as nn
import torch.optim as optim

import os
import data.utils as utils
from models import *

class Refiner(BaseModel):
    eps = 1e-7

    def __init__(self, opt):
        super(Refiner, self).__init__(opt)
        self.initialize()


    def initialize(self):
        self.opt_model = 'refiner'
        self.device = torch.device(self.device)

        self.guide_cnn = resnet50(pretrained_model=self.opt.resnet_path).to(self.device).eval()
        self.set_requires_grad(self.guide_cnn, False)

        self.generator = MultiScaleGenerator(self.opt.resnet_path).to(self.device)
        self.discriminator = CustomDiscriminator(in_channels=6, ndf=96).to(self.device)

        if not self.opt.eval:
            self.optimizer_G = optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
            self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

            self.criterion_GAN = networks.GANLoss(self.opt.gan_mode).to(self.device)
            self.criterion_DR = networks.DisRegulationLoss().to(self.device)
            self.criterion_L1 = nn.L1Loss().to(self.device)

            self.optimizers = [self.optimizer_G, self.optimizer_D]
        self.models = {'generator': self.generator, 'discriminator': self.discriminator}
        self.setup()


    def read_input(self, input):
        self.x = input['draft'].to(self.device)
        self.reference = input['reference'].to(self.device)
        self.y = input['color'].to(self.device)
        self.x_idx = input['index']


    def forward(self):
        features_r = self.guide_cnn(self.reference)
        self.predict_y = self.generator(self.x, features_r)


    def backward_D(self):
        fake_AB = torch.cat((self.x, self.predict_y), 1)
        predict_fake = self.discriminator(fake_AB.detach())
        self.loss_D_fake = self.criterion_GAN(predict_fake, False)

        true_AB = torch.cat((self.x, self.y), 1)
        predict_true = self.discriminator(true_AB)
        self.loss_D_true = self.criterion_GAN(predict_true, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_true) * 0.5
        self.loss_D.backward()


    def backward_G(self):
        fake_AB = torch.cat((self.x, self.predict_y), 1)
        predict_fake = self.discriminator(fake_AB)
        self.loss_GAN = self.criterion_GAN(predict_fake, True)
        self.loss_L1 = self.criterion_L1(self.predict_y, self.y) * self.opt.lambda_L1
        self.loss_DR = self.criterion_DR(self.predict_y, self.reference) * self.opt.lambda_dr

        self.loss_G = self.loss_GAN + self.loss_L1 + self.loss_DR
        self.loss_G.backward()


    def train(self):
        self.step = self.opt.start_step
        if self.opt.load_model:
            self.load(self.opt.load_epoch)

        for epoch in range(self.st_epoch, self.ed_epoch):
            for idx, data in enumerate(self.datasets):
                self.read_input(data)
                self.forward()

                # update discriminator
                self.set_requires_grad(self.discriminator, True)
                self.optimizer_D.zero_grad()
                self.backward_D()
                self.optimizer_D.step()

                # update refinement drawer
                self.set_requires_grad(self.discriminator, False)
                self.optimizer_G.zero_grad()
                self.backward_G()
                self.optimizer_G.step()

                if self.step % self.opt.print_state_freq == 0:
                    self.set_state_dict()
                    self.print_training_iter(epoch, idx)
                if self.step % self.opt.save_model_freq_step == 0:
                    self.output_samples(epoch, idx)
                    self.save()

                self.step += 1

            self.save()
            self.set_state_dict()
            self.print_training_epoch(epoch)

            if epoch % self.opt.save_model_freq == 0:
                self.save(epoch)


    def set_state_dict(self, eval=False):
        self.state_dict = {'loss_D': self.loss_D, 'loss_D_true': self.loss_D_true, 'loss_D_fake': self.loss_D_fake,
                           'loss_G': self.loss_G, 'loss_GAN': self.loss_GAN, 'loss_L1': self.loss_L1, 'loss_DR': self.loss_DR}


    def output_samples(self, epoch, index):
        self.generator.eval()

        with torch.no_grad():
            self.forward()

            utils.save_image(self.x[0].cpu(), os.path.join(self.sample_path, '{}_{}_input.png'.format(epoch, index)))
            utils.save_image(self.reference[0].cpu(), os.path.join(self.sample_path, '{}_{}_ref.png'.format(epoch, index)))
            utils.save_image(self.y[0].cpu(), os.path.join(self.sample_path, '{}_{}_real.png'.format(epoch, index)))
            utils.save_image(self.predict_y[0].cpu(), os.path.join(self.sample_path, '{}_{}_fake.png'.format(epoch, index)))

        self.generator.train()


    def test(self, load_epoch='latest'):
        self.generator.eval()

        with torch.no_grad():
            self.load()
            data_size = len(self.datasets)

            for idx, data in enumerate(self.datasets):
                self.read_input(data)
                self.forward()

                utils.save_image(self.x.squeeze(0).cpu(), os.path.join(self.test_path, load_epoch+'_{}_input.png'.format(self.x_idx[0])))
                utils.save_image(self.reference.squeeze(0).cpu(), os.path.join(self.test_path, load_epoch+'_{}_ref.png'.format(self.x_idx[0])))
                utils.save_image(self.y.squeeze(0).cpu(), os.path.join(self.test_path, load_epoch+'_{}_real.png'.format(self.x_idx[0])))
                utils.save_image(self.predict_y.squeeze(0).cpu(), os.path.join(self.test_path, load_epoch+'_{}_fake.png'.format(self.x_idx[0])))

                print('test proces: [{} / {}] ...'.format(idx+1, data_size))