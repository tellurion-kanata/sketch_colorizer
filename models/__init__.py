from .basemodel import *
from .pytorch_cnn import *
from .networks import *
from torch.nn import init

import torch.cuda


__all__ = [
    'BaseModel', 'define_R', 'define_G', 'define_M', 'define_D', 'loss', 'parallel', 'latent_shuffle'
]

""" GAN Loss, parallel training and initialization are from Jun-Yan Zhu et al.'s implementation 
Github: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 
"""

""" Applying zero normalization to generator. """
def init_net(net, init_gain=0.02, init_type='normal', gpus=[]):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    net = parallel(net, gpus)
    net.apply(init_func)
    return net


def parallel(net, gpus):
    if len(gpus) > 0:
        assert(torch.cuda.is_available())
        net = nn.DataParallel(net, gpus).to(gpus[0])
    return net

def define_attn_layer(attn_type, channels=512):
    if attn_type == 'fc':
        attn_layer = Attention(channels, channels)
    elif attn_type == 'se':
        attn_layer = SEAttention(channels, channels)
    elif attn_type == 'add':
        attn_layer = None
    else:
        raise NotImplementedError('Attention module [%s] is not found.' % attn_type)
    return attn_layer


def define_G(ngf, ref_channels=512, attn_type='fc', upsample_type='subpixel', attn_upsample=True, gpus=[]):
    attn_layer = define_attn_layer(attn_type)

    net = AttentionGenerator(
        ngf             = ngf,
        refer_ch        = ref_channels,
        attn_layer      = attn_layer,
        upsample_type   = upsample_type,
        atup            = attn_upsample,
    )

    net = init_net(net, gpus=gpus)
    return net

def define_M(input_ch, output_ch, n_layers, gpus=[]):
    net = MappingNetwork(
        input_ch    = input_ch,
        output_ch   = output_ch,
        n_layers    = n_layers,
    )

    net = init_net(net, gpus=gpus)
    return net


def define_D(input_ch, n_layers, ndf, spec_norm=False, gpus=[], ):
    net = CustomDiscriminator(
        in_channels = input_ch,
        n_layers    = n_layers,
        ndf         = ndf,
        spec_norm   = spec_norm,
    )

    net = init_net(net, init_type='kaiming', gpus=gpus)
    return net


def define_R(model, return_mode, pretrained=None, gpus=[]):
    try:
        net = getattr(pytorch_cnn, model)()
    except:
        raise NotImplementedError('Model [%s] is not found.' % model)

    net.set_return_mode(return_mode)
    if pretrained is not None:
        net.load_pretrained_model(torch.load(pretrained))

    ref_channels = 512 if model == 'resnet34' else 2048

    if len(gpus) > 0:
        assert(torch.cuda.is_available())
        net = nn.DataParallel(net, gpus).to(gpus[0])
    return net.eval(), ref_channels