import torch
import torch.nn as nn
import torch.nn.functional as functional

from torch.nn import Parameter

def latent_shuffle(x, is_train=False):
    if is_train:
        b, c, h, w = x.shape
        n = h * w
        x = x.view(b, c, n).permute(0, 2, 1)

        # shuffle the latent code
        new_ind = torch.randperm(n)
        replaced = x[:, new_ind]
        return replaced.permute(0, 2, 1).view(b, c, h, w).contiguous().detach()
    return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def DisConvLayer(in_channels, out_channels, kernel_size, stride, padding, spec_norm, bias):
    if spec_norm:
        conv = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    return conv


""" 
    ViT Implementation 
    Fully-connected attention module part 
    Github: https://github.com/lucidrains/vit-pytorch 
"""
class BaseAttention(nn.Module):
    def __init__(self, input_ch, ref_ch, use_bias=False, mode='spatial'):
        super(BaseAttention, self).__init__()
        assert mode in ['spatial', 'channel']
        self.q_w = nn.Linear(input_ch, input_ch, bias=use_bias)
        self.k_w = nn.Linear(ref_ch, input_ch, bias=use_bias)
        self.v_w = nn.Linear(ref_ch, input_ch, bias=use_bias)
        self.out_w = nn.Linear(input_ch, input_ch, bias=use_bias)

        self.scale = input_ch ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.spatial = mode == 'spatial'

    def compute_attn(self, q, k ,v):
        k = k.permute(0, 2, 1)
        if self.spatial:
            dots = torch.bmm(q, k) * self.scale
            attn = self.attend(dots)
            y = torch.bmm(attn, v)
        else:
            dots = torch.bmm(k, q) * self.scale
            attn = self.attend(dots)
            y = torch.bmm(v, attn)
        y = self.out_w(y)
        return y


class SEModule(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.s_w = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.e_w = nn.Conv2d(channels // reduction, channels, kernel_size=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        x = x.mean(dim=[2, 3], keepdims=True)
        x = self.s_w(x)
        x = self.relu(x)
        x = self.e_w(x)
        x = self.sigmoid(x)

        return identity * x


class Attention(BaseAttention):
    def __init__(self, input_ch, ref_ch, use_bias=False, mode='spatial'):
        super(Attention, self).__init__(input_ch, ref_ch, use_bias, mode)

    def forward(self, x, fr):
        b, c, h, w = x.shape
        x = x.view(b, -1, h * w).permute(0, 2, 1)
        fr = fr.view(b, -1, h * w).permute(0, 2, 1)

        q = self.q_w(x)
        k = self.k_w(fr)
        v = self.v_w(fr)
        attn = self.compute_attn(q, k, v)
        attn = attn.permute(0, 2, 1).view(b, -1, h, w)

        return attn


class SEAttention(BaseAttention):
    def __init__(self, input_ch, ref_ch, reduction=8):
        super(SEAttention, self).__init__(input_ch, ref_ch)
        self.se = SEModule(input_ch, reduction)

    def forward(self, x, fr):
        b, c, h, w = x.shape
        x = x.view(b, -1, h * w).permute(0, 2, 1)
        fr = fr.view(b, -1, h * w).permute(0, 2, 1)

        q = self.q_w(x)
        k = self.k_w(fr)
        v = self.v_w(fr)
        attn = self.compute_attn(q, k, v)
        attn = attn.permute(0, 2, 1).view(b, -1, h, w)
        attn = self.se(attn)

        return attn


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, relu):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = relu

    def forward(self, x):
        residual = x
        y = self.relu(self.conv1(x))
        y = self.conv2(y)
        y = y + residual
        y = self.relu(y)

        return y


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = relu

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        return y


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class UpsampleConvBlock(nn.Module):
    """Based on UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, relu, attn_up=False, ref_channels=512, final=False):
        super(UpsampleConvBlock, self).__init__()
        self.reflection_pad = nn.ReflectionPad2d(1)

        mid_channels = out_channels
        out_channels = out_channels if not final else 3
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.relu = relu
        self.final = final

        self.attn_layer = None
        if attn_up:
            self.attn_layer = Attention(out_channels, ref_channels, mode='channel')
            self.sigmoid = nn.Sigmoid()


    def forward(self, x, x_skip=None, hint=None):
        # _, c, h, w = x_skip.shape
        # feature_r = functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)     # Activate interpolate during metric in case the input size is not a multiple of 32
        if x_skip is not None:
            x = torch.cat((x, x_skip), 1)
        x = functional.interpolate(x, mode='nearest', scale_factor=2)
        x = self.reflection_pad(x)
        out = self.relu(self.conv1(x))

        if self.attn_layer:
            out = out + out * self.sigmoid(self.attn_layer(out.mean(dim=[2, 3], keepdims=True), hint))
        out = self.conv2(out)

        if self.final:
            out = self.relu(out)
        return out


class SubPixelUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, attn_up=False, relu=nn.ReLU(), ref_channels=512, final=False):
        super(SubPixelUpsampleBlock, self).__init__()
        mid_channels = out_channels
        out_channels = out_channels if not final else 3
        self.conv1 = nn.Conv2d(in_channels, mid_channels * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(2)
        self.relu = relu
        self.final = final

        self.attn_layer = None
        if attn_up:
            self.attn_layer = Attention(out_channels, ref_channels, mode='channel')
            self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_skip=None, hint=None):
        if x_skip is not None:
            x = torch.cat((x, x_skip), 1)
        out = self.relu(self.conv1(x))
        out = self.upsample(out)

        if self.attn_layer:
            out = out + out * self.sigmoid(self.attn_layer(out.mean(dim=[2, 3], keepdims=True), hint))
        out = self.conv2(out)

        if not self.final:
            out = self.relu(out)
        return out


class AttentionUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, relu):
        super(AttentionUpBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.sattn = Attention(out_channels, out_channels)
        self.upsample = nn.PixelShuffle(2)
        self.relu = relu


    def forward(self, x, x_skip=None):
        if x_skip is not None:
            x = torch.cat((x, x_skip), 1)
        out = self.relu(self.conv(x))
        out = self.upsample(out)
        out = out + self.sattn(out, out)
        return out