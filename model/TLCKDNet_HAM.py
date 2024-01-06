import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .rest import ResT as Pre_Rest
import os
import torch.nn.functional as F
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 apply_transform=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, apply_transform=apply_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class GL(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gl_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        return x + self.gl_conv(x)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(self, patch_size=16, in_ch=3, out_ch=768, with_pos=False):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size + 1, stride=patch_size, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(out_ch)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class BasicStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x


class Stem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(Stem, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.max_pool(x)

        if self.with_pos:
            x = self.pos(x)
        return x


class ResT(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
                 norm_layer=nn.LayerNorm, apply_transform=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.apply_transform = apply_transform

        self.stem = BasicStem(in_ch=in_chans, out_ch=embed_dims[0], with_pos=True)

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True)
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True)
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform)
            for i in range(self.depths[1])])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform)
            for i in range(self.depths[2])])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform)
            for i in range(self.depths[3])])

        self.norm = norm_layer(embed_dims[3])

        # classification head
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x1 = x

        # stage 2
        x, (H, W) = self.patch_embed_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x2 = x

        # stage 3
        x, (H, W) = self.patch_embed_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        x3 = x

        # stage 4
        x, (H, W) = self.patch_embed_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)

        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        # x = self.avg_pool(x).flatten(1)
        # x = self.head(x)
        return x1, x2, x3, x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y
class HAM(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(HAM, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = ChannelAttention(dim, reduction=8)
        self.salayer = SpatialAttention(kernel_size=7)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        ca_res = self.calayer(res)
        sa_res = self.salayer(res)
        res = ca_res + sa_res
        res = self.palayer(res)
        res += x

        return res

class MogLKDWConv2d(nn.Module):
    def __init__(self, in_channel, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        gc = int(in_channel * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = (in_channel - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat([x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)], dim=1)

class GoolgNetDWConv(nn.Module):
    def __init__(self, in_channel, branch_kernel_size=[7, 11, 21], split_ratio=0.125):
        super().__init__()
        cs = int(in_channel * split_ratio)
        self.dwconv_hw = nn.Conv2d(cs, cs, branch_kernel_size[0], padding=branch_kernel_size[0] // 2, groups=cs)
        self.dwconv_w1 = nn.Conv2d(cs, cs, kernel_size=(1, branch_kernel_size[1]),
                                   padding=(0, branch_kernel_size[1] // 2), groups=cs)
        self.dwconv_h1 = nn.Conv2d(cs, cs, kernel_size=(branch_kernel_size[1], 1),
                                   padding=(branch_kernel_size[1] // 2, 0), groups=cs)
        self.dwconv_w2 = nn.Conv2d(cs, cs, kernel_size=(1, branch_kernel_size[1]),
                                   padding=(0, branch_kernel_size[1] // 2), groups=cs)
        self.dwconv_h2 = nn.Conv2d(cs, cs, kernel_size=(branch_kernel_size[1], 1),
                                   padding=(branch_kernel_size[1] // 2, 0), groups=cs)
        self.dwconv_hw3 = nn.Conv2d(cs, cs, kernel_size=3, padding=1, groups=cs)

        self.split_size = (in_channel - 4 * cs, cs, cs, cs, cs)

    def forward(self, x):
        x_ide, x_m, x_hw, x_hw1, x_hw2 = torch.split(x, self.split_size, dim=1)
        x_hw = self.dwconv_hw(x_hw)
        x_hw1 = self.dwconv_h1(self.dwconv_w1(x_hw1))
        x_hw2 = self.dwconv_h2(self.dwconv_w2(x_hw2))
        x_hw3 = self.dwconv_hw3(x_m)
        return torch.cat([x_ide, x_hw3, x_hw, x_hw1, x_hw2], dim=1)


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    copied from timm: https://github.com/huggingface/pytorch-image-models/blob/v0.6.11/timm/models/layers/mlp.py
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU,
            norm_layer=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x



class MogRM(nn.Module):
    """ MetaNeXtBlock Block
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            dim,
            token_mixer=GoolgNetDWConv,
            norm_layer=nn.BatchNorm2d,
            mlp_layer=ConvMlp,
            mlp_ratio=4,
            act_layer=nn.GELU,
            ls_init_value=1e-6,
            drop_path=0.,

    ):
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(dim)) if ls_init_value else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

class Decoder_Transformer(nn.Module):
    def __init__(self, dim=[768,384,192,96], heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., act_layer=nn.GELU, proj_drop=0., drop=0.):
        super().__init__()
        self.num_heads = heads
        head_dim = dim[1] // heads
        self.scale = qk_scale or head_dim ** -0.5
        #self.scale = qk_scale or dim[1] // heads ** -0.5

        self.q = nn.Sequential(nn.Conv2d(dim[0], dim[1], kernel_size=1),
                               #DMConv(dim[1]),
                               MogLKDWConv2d(dim[1], square_kernel_size=3, band_kernel_size=21, branch_ratio=0.125),
                               nn.UpsamplingBilinear2d(scale_factor=2)
                               )
        self.k = nn.Sequential(
                               #DMConv(dim[1])
            MogLKDWConv2d(dim[1], square_kernel_size=7, band_kernel_size=21, branch_ratio=0.125),
                               )
        self.v = nn.Sequential(nn.Conv2d(dim[2], dim[1], kernel_size=1),
                               #DMConv(dim[1])
                               MogLKDWConv2d(dim[1], square_kernel_size=3, band_kernel_size=21, branch_ratio=0.125),
                               )
        self.attn_drop = nn.Dropout(attn_drop)
        self.identity = nn.Sequential(nn.Conv2d(dim[1], dim[3], kernel_size=1),
                                      #DMConv(dim[3]),
                                      MogLKDWConv2d(dim[3], square_kernel_size=3, band_kernel_size=21,
                                                         branch_ratio=0.125),
                                      nn.UpsamplingBilinear2d(scale_factor=2))
        self.proj = nn.Conv2d(dim[3], dim[3], kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv = nn.Sequential(nn.Conv2d(180, 96, kernel_size=1)
                                  #nn.ReLU(inplace=True)
                                  #nn.GELU()
                                  )
        self.up8 = nn.PixelShuffle(8)
        self.up4 = nn.PixelShuffle(4)
        self.up2 = nn.PixelShuffle(2)

        mlp_hidden_dim = int(dim[3] * 4)
        self.norm = nn.LayerNorm(dim[3])
        self.MLP = Mlp(in_features=dim[3], hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x1, x2, x3, x4):
        B, _, H, W = x2.shape
        q = rearrange(self.q(x4), 'b (c head) h w -> b head (h w) c', head=self.num_heads)
        #feature = q
        k = rearrange(self.k(x3), 'b (c head) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(self.v(x2), 'b (c head) h w -> b head (h w) c', head=self.num_heads)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        #feature = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #feature = attn
        attn = F.interpolate(attn, scale_factor=4, mode='bilinear', align_corners=True)
        #feature = attn


        x = (attn @ v).transpose(1, 2)
        x = rearrange(x, 'b (h w) head c -> b (c head) h w', head=self.num_heads, h=H, w=W)
        x = self.identity(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x1 = self.conv(x1)
        x4 = self.up8(x4)
        x3 = self.up4(x3)
        x2 = self.up2(x2)
        x1 = torch.cat([x1, x2, x3, x4], dim=1)
        x1 = self.conv(x1)
        x = x + x1
        feature = x
        H1, W1 = x.shape[2:]
        x_ = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x_)
        x = self.MLP(x)
        x = x + x_
        x = rearrange(x, 'b (h w) c -> b c h w', h=H1, w=W1)
        return x, feature


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class TLCKDNet(nn.Module):
    def __init__(self, cfg):
        # def __init__(self):
        super(TLCKDNet, self).__init__()
        self.encoder = ResT(embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                            qkv_bias=True,
                            depths=[2, 2, 18, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True)
        rest = Pre_Rest(embed_dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                              qkv_bias=True,
                              depths=[2, 2, 18, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True)
        rest.load_state_dict(torch.load(os.path.join(cfg.imagenet_model, 'rest_large.pth')))
        pretrained_dict = rest.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)

        self.decoder = Decoder_Transformer(dim=[768,384,192,96], heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
                                           act_layer=nn.GELU, proj_drop=0., drop=0.)

        self.MogRM = nn.Sequential(
                                 #MogRM(192),
                                 nn.Conv2d(192, 96, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(96),
                                 nn.ReLU(inplace=True))

        self.HAM = nn.Identity()

        self.upsample2_ = nn.UpsamplingBilinear2d(scale_factor=2)

        self.output = nn.Sequential(nn.Conv2d(96, 32, kernel_size=3, padding=1),
                                    nn.ReLU(True),
                                    nn.Conv2d(32, 1, kernel_size=3, padding=1))

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)

        x_dec, f = self.decoder(x1, x2, x3, x4)

        x_sum1 = self.MogRM(self.upsample2_(torch.cat([x_dec, x1],dim=1)))

        x_sum = self.HAM(self.upsample2_(x_sum1))

        y = self.output(x_sum)

        return torch.sigmoid(y)


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis
    img = torch.randn(1, 3, 256, 256)
    model = TLCKDNet('../ImgNet')
    flops = FlopCountAnalysis(model, img)
    print('flops: {}'.format(flops.total() / 1e9))
    y = model(img)
    print(y.shape)



