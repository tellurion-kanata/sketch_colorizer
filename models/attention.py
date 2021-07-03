import torch
import torch.nn as nn

from einops import rearrange

""" ViT Implementation """
""" Github: https://github.com/lucidrains/vit-pytorch """

class Attention(nn.Module):
    def __init__(self, input_ch, heads=8, dropout=0.):
        super(Attention, self).__init__()
        self.heads = heads
        ch_head = input_ch // heads
        self.scale = ch_head ** -0.5

        inner_ch = heads * ch_head
        self.attend = nn.Softmax(dim=-1)
        self.qkv_w = nn.Linear(input_ch, inner_ch*3, bias=False)

        self.out_w = nn.Linear(inner_ch, input_ch // 2)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        qkv = self.qkv_w(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, ('b n (h c) -> b h n c'), h=h), qkv)

        dots = torch.einsum('b h i c, b h j c -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        y = torch.einsum('b h i j, b h j c -> b h i c', attn, v)
        y = rearrange(y, 'b h n c -> b n (h c)')

        y = self.out_w(y)
        y = self.dropout(y)
        return y