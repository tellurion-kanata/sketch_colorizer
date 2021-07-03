import torch
import torch.nn as nn

from models.resnet import resnet50
from .attention import *

""" GAN Loss implemented by Jun-Yan Zhu et al. 
Github: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
"""

class SelfRegulationLoss(nn.Module):
    def __init__(self):
        super(SelfRegulationLoss, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h*w)
        loss = -x.var(dim=2)
        return loss


class DisRegulationLoss(nn.Module):
    def __init__(self):
        super(DisRegulationLoss, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, x, y):
        b, c, h, w = x.shape
        x, y = x.view(b, c, h*w), y.view(b, c, h*w)
        loss = self.loss(x.var(dim=2), y.var(dim=2))

        return loss


class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.relu(self.conv1(out))
        out = self.conv2(out)

        return out


class AttentionUpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(AttentionUpsampleConvLayer, self).__init__()
        self.upsample = upsample
        padding = kernel_size // 2
        self.mhsa = Attention(in_channels*2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU()


    def attn_forward(self, x, features_r):
        _, c, h, w = x.shape
        feature = torch.cat((x, features_r), 1).view(-1, c*2, h * w).permute(0, 2, 1)
        attn = self.mhsa(feature)
        attn = x + attn.view(-1, c, h, w)
        return attn


    def forward(self, x, feature, y=None):
        feature = self.attn_forward(x, feature)
        x_in = y+feature if y is not None else feature
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.relu(self.conv1(x_in))
        out = self.conv2(out)

        return out



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        return y


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        y = y + residual
        return y


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Convolution Encoder
        self.conv1 = ConvBlock(3, 128, kernel_size=3)                            # 256
        self.conv2 = ConvBlock(128, 256, kernel_size=3)                          # 128
        self.conv3 = ConvBlock(256, 512, kernel_size=3)                         # 64
        self.conv4 = ConvBlock(512, 1024, kernel_size=3)                         # 32
        # Residual Blocks
        self.res1 = ResidualBlock(1024)
        self.res2 = ResidualBlock(1024)
        # Multi-head self-attention Layer
        self.mhsa = Attention(2048)
        # Convolution Decoder
        self.deconv1 = UpsampleConvLayer(1024, 512, kernel_size=3, stride=1, upsample=2)    # 64
        self.deconv2 = UpsampleConvLayer(512, 256, kernel_size=3, stride=1, upsample=2)     # 128
        self.deconv3 = UpsampleConvLayer(256, 128, kernel_size=3, stride=1, upsample=2)     # 256
        self.deconv4 = UpsampleConvLayer(128, 3, kernel_size=3, stride=1, upsample=2)       # 512
        self.tanh = nn.Tanh()
        # Non-Linearity
        self.relu = nn.ReLU()

        self.encoder = nn.Sequential(*[
            self.conv1, self.relu,
            self.conv2, self.relu,
            self.conv3, self.relu,
            self.conv4, self.relu
        ])

        self.residual = nn.Sequential(*[
            self.res1, self.res2
        ])

        self.attention = nn.Sequential(*[
            self.mhsa
        ])

        self.decoder = nn.Sequential(*[
            self.deconv1, self.relu,
            self.deconv2, self.relu,
            self.deconv3, self.relu,
            self.deconv4, self.tanh
        ])


    def attn_layer(self, x, feature):
        _, c, h, w = x.shape
        feature = torch.cat((x, feature), 1).view(-1, c*2, h * w).permute(0, 2, 1)
        attn = self.attention(feature)
        attn = attn.view(-1, c, h, w)
        out = x + attn
        return out


    def forward(self, x, feature_r):
        y = self.encoder(x)
        y = self.residual(y)

        y = self.attn_layer(y, feature_r)                # [1, 1024, 32, 32] -> [1, 512, 32, 32]

        y = self.decoder(y)
        return y


class MultiScaleGenerator(nn.Module):
    def __init__(self, pretrained_resnet):
        super(MultiScaleGenerator, self).__init__()
        self.encoder = resnet50(pretrained_model=pretrained_resnet)

        self.deconv1 = AttentionUpsampleConvLayer(2048, 1024, kernel_size=3, stride=1, upsample=2)     # 32
        self.in1 = nn.InstanceNorm2d(1024, affine=True)

        self.deconv2 = AttentionUpsampleConvLayer(1024, 512, kernel_size=3, stride=1, upsample=2)    # 64
        self.in2 = nn.InstanceNorm2d(512, affine=True)

        self.deconv3 = AttentionUpsampleConvLayer(512, 256, kernel_size=3, stride=1, upsample=2)     # 128
        self.in3 = nn.InstanceNorm2d(256, affine=True)

        self.deconv4 = AttentionUpsampleConvLayer(256, 64, kernel_size=3, stride=1, upsample=2)      # 256
        self.in4 = nn.InstanceNorm2d(64, affine=True)

        self.deconv5 = UpsampleConvLayer(64, 3, kernel_size=7, stride=1, upsample=2)       # 512
        self.tanh = nn.Tanh()
        self.relu = nn.GELU()


    def forward(self, x, features_r):
        features_x = self.encoder(x)
        y = self.relu(self.deconv1(features_x[3], features_r[3]))
        y = self.relu(self.deconv2(features_x[2], features_r[2], y))
        y = self.relu(self.deconv3(features_x[1], features_r[1], y))
        y = self.relu(self.deconv4(features_x[0], features_r[0], y))

        y = self.deconv5(y)
        y = self.tanh(y)

        return y


""" MultiLayerDiscriminator implemented by Jun-Yan Zhu et al. 
Github: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
"""
class CustomDiscriminator(nn.Module):
    def __init__(self, in_channels, n_layers=3, ndf=64, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(CustomDiscriminator, self).__init__()
        model = []
        model += [
            nn.Conv2d(in_channels, ndf, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(8, 2 ** n)
            model += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        model += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1)]
        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)