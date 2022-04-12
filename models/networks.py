from .modules import *
from functools import partial


class EncoderDecoder(nn.Module):
    def __init__(self, ngf=64, upsample_type='subpixel', atup=False):
        super(EncoderDecoder, self).__init__()
        if upsample_type == 'subpixel':
            UpsampleBlock = SubPixelUpsampleBlock
        elif upsample_type == 'interpolate':
            UpsampleBlock = UpsampleConvBlock
        else:
             raise NotImplementedError('Upsample layer {} is not implemented.'.format(upsample_type))

        # Activation function
        self.relu = nn.ReLU(inplace=True)
        DownConv = partial(ConvBlock, relu=self.relu)
        UpConv = partial(UpsampleBlock, relu=self.relu)

        # Convolution Encoder
        self.conv1 = DownConv(1, ngf)                                        # H
        self.conv2 = DownConv(ngf, ngf*2)                                    # H/2
        self.conv3 = DownConv(ngf*2, ngf*4)                                  # H/4
        self.conv4 = DownConv(ngf*4, ngf*8)                                  # H/8
        self.conv5 = DownConv(ngf*8, 512)                                    # H/16

        self.res = ResidualBlock(512, self.relu)
        self.in_conv = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        # Convolution Decoder
        self.deconv1 = AttentionUpBlock(ngf*8, ngf*8, self.relu)             # H/16
        self.deconv2 = UpConv(ngf*16, ngf*4, atup)                           # H/8
        self.deconv3 = UpConv(ngf*8, ngf*2, atup)                            # H/4
        self.deconv4 = UpConv(ngf*4, ngf, atup)                              # H/2
        self.deconv5 = UpConv(ngf*2, ngf, final=True)                        # H
        self.tanh = nn.Tanh()


class AttentionGenerator(EncoderDecoder):
    def __init__(self, ngf=64, refer_ch=512, attn_layer=None, upsample_type='subpixel', atup=False):
        super(AttentionGenerator, self).__init__(ngf, upsample_type, atup)

        # Residual Blocks
        self.refer_attn = attn_layer

        self.atup = atup
        self.downsample = nn.Conv2d(refer_ch, 512, kernel_size=1) if refer_ch > 512 else None

    def forward(self, x, fr, hook=False):
        fr = self.downsample(fr) if self.downsample else fr
        hint = fr.mean(dim=[2, 3], keepdims=True) if self.atup else None

        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y = self.conv5(y4)

        y = self.res(y)

        if self.refer_attn:
            y = y + self.refer_attn(y, fr)
        else:
            y = y + fr.mean(dim=[2, 3], keepdims=True)
        y = self.relu(self.in_conv(y))
        fy = y if hook else None

        y = self.deconv1(y)
        y = self.deconv2(y, y4, hint)
        y = self.deconv3(y, y3, hint)
        y = self.deconv4(y, y2, hint)
        y = self.deconv5(y, y1)

        y = self.tanh(y)
        return y, fy


class MappingNetwork(nn.Module):
    def __init__(self, input_ch, output_ch, n_layers=4):
        super(MappingNetwork, self).__init__()
        act_func = nn.ReLU(inplace=True)
        self.bch = output_ch
        self.fch = output_ch * n_layers

        model = [nn.Sequential(*[nn.Linear(input_ch, output_ch), act_func])]
        for i in range(1, n_layers):
            model += [nn.Sequential(*[nn.Linear(output_ch, output_ch), act_func])]
        self.mlp = nn.Sequential(*model)
        self.fusion_layer = nn.Linear(self.fch, output_ch, bias=False)

    def forward(self, x):
        b, _ = x.shape

        fx = torch.zeros([b, self.fch], device=x.device)
        for idx, layer in enumerate(self.mlp):
            x = layer(x)
            fx[:, idx * self.bch: (idx+1) * self.bch] = x
        out = self.fusion_layer(fx)
        return out.view(b, -1, 1, 1)


""" Default stage Discriminator
Downsampling factor: 2^n_layer + 1 (downsample = True)   defalut is 3
                     2^n_layer (downsample = False)
"""
class CustomDiscriminator(nn.Module):
    def __init__(self, in_channels, n_layers=3, ndf=64, downsample=False, spec_norm=False, use_bias=True):
        super(CustomDiscriminator, self).__init__()
        model = []
        model += [
            DisConvLayer(in_channels, 64, kernel_size=3, stride=1, padding=1, spec_norm=spec_norm, bias=use_bias),
            nn.LeakyReLU(0.2, True),
            DisConvLayer(64, 64, kernel_size=3, stride=2, padding=1, spec_norm=spec_norm, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        prev = 64
        for n in range(1, n_layers):
            nf_mult = min(2 ** n, 8)
            model += [
                DisConvLayer(prev, ndf * nf_mult, kernel_size=3, stride=1, padding=1, spec_norm=spec_norm, bias=use_bias),
                nn.LeakyReLU(0.2, True),
                DisConvLayer(ndf * nf_mult, ndf * nf_mult, kernel_size=3, stride=2, padding=1, spec_norm=spec_norm, bias=use_bias),
                nn.LeakyReLU(0.2, True)
            ]
            prev = nf_mult * ndf

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        stride = 2 if downsample else 1
        model += [
            DisConvLayer(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=stride, padding=1, spec_norm=spec_norm, bias=use_bias),
            nn.LeakyReLU(0.2, True)
        ]

        model += [DisConvLayer(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=1, spec_norm=spec_norm, bias=use_bias)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)