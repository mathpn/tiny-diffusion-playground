"""
Latent Difussion Model (self-contained).
"""

# pylint: disable=no-member
import functools
import hashlib
import os
from collections import namedtuple

import requests
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from torchvision.datasets import Flowers102
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        norm_layer = nn.BatchNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, disc_conditional=False):

        super().__init__()
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                )
        self.disc_loss = hinge_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, z, optimizer_idx,
                last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        # weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        weighted_nll_loss = weighted_nll_loss.mean()
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = nll_loss.mean()
        kl_loss = kl_div(z.mean(), z.var().log())

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * self.disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(self.disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            d_loss = self.disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        out = self.conv2(x1)
        if self.same_channels:
            out = out + x
        else:
            out = out + x1
        return out


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


def kl_div(mean1, logvar1, mean2 = torch.tensor(0.), logvar2 = torch.tensor(1.).log()):
    mean2 = mean2.to(device)
    logvar2 = logvar2.to(device)
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        channel_multiplier: list[int],
        n_res_blocks: int = 1,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, 3, 1, 1)
        in_channel_multiplier = [1] + list(channel_multiplier)
        down = []
        for i, mult in enumerate(channel_multiplier):
            block = []
            block_in = channels * in_channel_multiplier[i]
            block_out = channels * mult
            for _ in range(n_res_blocks):
                block.append(ResidualBlock(block_in, block_out))
                block_in = block_out
            if i != len(channel_multiplier) - 1:
                block.append(Downsample(block_in, True))
            down.append(nn.Sequential(*block))
        self.down = nn.Sequential(*down)

        block_in = channels * channel_multiplier[-1]
        self.middle = ResidualBlock(block_in, block_in)
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down(x)
        x = self.middle(x)
        x = self.norm_out(x)
        x = self.conv_out(x)
        return 2 * torch.tanh(x)  # [-2, 2] to allow KL divergence to work


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        channel_multiplier: list[int],
        *args,
        **kwargs,
    ):
        super().__init__()
        block_in = channels * channel_multiplier[-1]
        self.conv_in = nn.Conv2d(out_channels, block_in, 3, 1, 1)

        self.middle = ResidualBlock(block_in, block_in)

        up = []
        num_resolutions = len(channel_multiplier)
        for i in reversed(range(num_resolutions)):
            block = []
            block_out = channels * channel_multiplier[i]
            block.append(ResidualBlock(block_in, block_out))
            block_in = block_out
            if i != 0:
                block.append(Upsample(block_in, True))
            up.append(nn.Sequential(*block))
        self.up = nn.Sequential(*up)

        block_in = channels
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, in_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.up(x)
        x = self.norm_out(x)
        x = self.conv_out(x)
        return torch.tanh(x)


class Autoencoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Decoder(*args, **kwargs)
        self.loss = LPIPSWithDiscriminator(disc_weight=0.5, kl_weight=1e-5, disc_num_layers=2)

    def forward(self, x):
        z = self.encoder(x)
        # print("encode")
        # print(((z + 2) / 4).min(), ((z + 2) / 4).max())
        out = self.decoder((z + 2) / 4)  # [0, 1]
        return out, z

    def encode(self, x):
        return (self.encoder(x) + 2) / 4  # [0, 1]

    def decode(self, z):
        # print("decode")
        # print(z.min(), z.max())
        return self.decoder(z)

    def training_step(self, batch, optimizer_idx):
        reconstructions, z = self(batch)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict = self.loss(batch, reconstructions, z, optimizer_idx,
                                            last_layer=self.get_last_layer(), split="train")
            return aeloss, log_dict

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict = self.loss(batch, reconstructions, z, optimizer_idx,
                                                last_layer=self.get_last_layer(), split="train")

            return discloss, log_dict

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def configure_optimizers(self, lr: float):
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc]


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = ResidualBlock(in_channels, out_channels)
        self.conv_2 = ResidualBlock(out_channels, out_channels)
        self.downsample = Downsample(out_channels)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return self.downsample(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = ResidualBlock(in_channels, out_channels)
        self.conv_2 = ResidualBlock(out_channels, out_channels)
        self.upsample = Upsample(out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return self.upsample(x)


class TimeSiren(nn.Module):
    # https://github.com/cloneofsimo/minDiffusion/blob/master/mindiffusion/unet.py
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim),
        )
        self.ln = LayerNorm(dim)

    def forward(self, x):
        x_norm = self.ln(x)
        b, c, h, w = x.shape
        qkv = self.to_qkv(x_norm).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out) + x


class UNet(nn.Module):
    # https://arxiv.org/abs/2006.11239
    def __init__(self, in_channels: int, out_channels: int, n_feat: int):
        super().__init__()
        self.n_feat = n_feat
        self.conv1 = ResidualBlock(2 * in_channels, n_feat)
        self.down1 = nn.Sequential(
            DownConv(n_feat, n_feat),
            LinearAttention(n_feat),
        )
        self.down2 = nn.Sequential(
            DownConv(n_feat, 2 * n_feat),
            LinearAttention(2 * n_feat),
        )
        self.down3 = nn.Sequential(
            DownConv(2 * n_feat, 4 * n_feat),
            LinearAttention(4 * n_feat),
        )
        self.down4 = nn.Sequential(
            DownConv(4 * n_feat, 8 * n_feat),
            LinearAttention(8 * n_feat),
        )

        self.timeembed = TimeSiren(8 * n_feat)
        self.middle_conv = nn.Sequential(
            ResidualBlock(8 * n_feat, 8 * n_feat),
            LinearAttention(8 * n_feat),
            ResidualBlock(8 * n_feat, 8 * n_feat),
        )

        self.up1 = UpConv(16 * n_feat, 4 * n_feat)
        self.att1 = LinearAttention(4 * n_feat)
        self.up2 = UpConv(8 * n_feat, 2 * n_feat)
        self.att2 = LinearAttention(2 * n_feat)
        self.up3 = UpConv(4 * n_feat, n_feat)
        self.att3 = LinearAttention(n_feat)
        self.up4 = UpConv(2 * n_feat, n_feat)
        self.att4 = LinearAttention(n_feat)
        self.final_block = ResidualBlock(2 * n_feat, n_feat)
        self.out_conv = nn.Conv2d(n_feat, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, x_self_cond=None) -> torch.Tensor:
        if x_self_cond is None:
            x_self_cond = torch.zeros_like(x)
        x = torch.cat((x, x_self_cond), dim=1)
        x = self.conv1(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        middle = self.middle_conv(d4)
        temb = self.timeembed(t).view(-1, self.n_feat * 8, 1, 1).repeat(1, 1, 2, 2)

        up0 = middle + temb
        up1 = self.up1(up0, d4)
        up1 = self.att1(up1)
        up2 = self.up2(up1, d3)
        up2 = self.att2(up2)
        up3 = self.up3(up2, d2)
        up3 = self.att3(up3)
        up4 = self.up4(up3, d1)

        out = self.final_block(torch.cat((up4, x), 1))
        return self.out_conv(out)


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: tuple[float, float],
        n_T: int,
    ):
        super().__init__()
        self.eps_model = eps_model
        self.n_T = n_T

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2 * x - 1  # [0, 1] to [-1, 1]
        ts = torch.randint(1, self.n_T + 1, x.shape[0:1]).to(x.device)
        eps = torch.randn_like(x)

        x_t = (
            self.sqrtab[ts, None, None, None] * x + self.sqrtmab[ts, None, None, None] * eps
            # self.sqrtab[ts, None] * x + self.sqrtmab[ts, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        eps_h = self.eps_model(x_t, ts / self.n_T)
        return (eps - eps_h).abs().mean() #+ 1e-3 * kl_div(eps_h.mean(), eps_h.var().log())

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        x_start = None
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            time = torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            eps = self.eps_model(x_i, time, x_start)
            # start from noise
            x_start = x_i - self.sqrtmab[i, None, None, None] * eps / self.sqrtab[i, None, None, None]
            x_start = x_start.clamp(-1, 1)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            x_i = x_i.clamp(-1, 1)

        x_i = (x_i + 1) / 2  # [-1, 1] to  [0, 1]
        return x_i


def ddpm_schedules(beta1: float, beta2: float, T: int) -> dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

from PIL import Image


class Dataset:
    def __init__(self, transform):
        self.files = os.listdir("./data/flowers-102/jpg")
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(Image.open(f"./data/flowers-102/jpg/{self.files[idx]}"))

    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    ae_epochs = 5
    diff_epochs = 100
    betas = (1e-4, 0.02)
    n_T = 500
    n_feat = 128
    accumulate_steps = 1
    device = "cuda:0"

    ae = Autoencoder(3, 3, 64, [1, 2, 4])
    print(f"Number of AE parameters: {sum(p.numel() for p in ae.parameters() if p.requires_grad)}")
    ae.to(device)
    # load pre-trained AE
    # state_dict = torch.load("./ae_flowers.pth")
    # ae.load_state_dict(state_dict)

    transform = T.Compose([
        T.Resize(128),
        T.CenterCrop(128),
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(128, (0.7, 1.0)),
        T.ToTensor(),
        # lambda tensor: 2 * tensor - 1,
    ])
    # unnormalize = lambda tensor: (tensor.clamp(min=-1, max=1) + 1) / 2
    unnormalize = lambda tensor: tensor
    # dataset = Flowers102("./data", split="train", download=True, transform=transform)
    dataset = Dataset(transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

    # train AE
    optim_ae, optim_disc = ae.configure_optimizers(1e-4)
    optims = [optim_ae, optim_disc]


    for epoch in range(ae_epochs):
        ae.train()

        pbar = tqdm(loader)
        ema_ae = None
        ema_disc = None
        for x in pbar:
            losses = []
            x = x.to(device)
            for optim_id, optim in enumerate(optims):
                optim.zero_grad()
                loss, log_dict = ae.training_step(x, optim_id)
                losses.append(loss)
                loss.backward()
                optim.step()
            if ema_ae is None:
                ema_ae = losses[0].item()
                ema_disc = losses[1].item()
            else:
                ema_ae = 0.9 * ema_ae + 0.1 * losses[0].item()
                ema_disc = 0.9 * ema_disc + 0.1 * losses[1].item()
            pbar.set_description(f"epoch {epoch + 1} - loss AE: {ema_ae:.3f} loss disc: {ema_disc:.2f}")

        ae.eval()
        with torch.no_grad():
            reco, hidden = ae(x[0:8])
            images = unnormalize(torch.cat((reco, x[0:8]), dim=0))
            grid = make_grid(images, nrow=4)
            save_image(grid, f"./contents/ae_sample_{epoch}.png")
            hidden = unnormalize(hidden / 2)
            grid = make_grid(hidden, nrow=4)
            save_image(grid, f"./contents/ae_hidden_{epoch}.png")

            # save model
            torch.save(ae.state_dict(), f"./ae_flowers.pth")

    from denoising_diffusion_pytorch import GaussianDiffusion, Trainer, Unet

    transform = T.Compose([
        T.Resize(128),
        T.CenterCrop(128),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    unnormalize = lambda tensor: tensor
    dataset = Dataset(transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

    unet = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    ).cuda()

    diffusion = GaussianDiffusion(
        unet,
        image_size = 32,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 500,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1'            # L1 or L2
    ).cuda()

    ddpm = diffusion
    print(f"Number of DDPM parameters: {sum(p.numel() for p in ddpm.parameters() if p.requires_grad)}")

    optim = torch.optim.Adam(ddpm.parameters(), lr=8e-5, betas=(0.9, 0.99))
    #scheduler = OneCycleLR(optim, max_lr=2e-4, total_steps=len(loader) * diff_epochs)
    for epoch in range(diff_epochs):
        ddpm.train()
        ae.eval()

        pbar = tqdm(loader)
        loss_ema = None
        total_loss = 0.
        for i, x in enumerate(pbar):
            if (i + 1) % accumulate_steps == 0:
                optim.step()
                #scheduler.step()
                optim.zero_grad()
                if loss_ema is None:
                    loss_ema = total_loss
                else:
                    loss_ema = 0.9 * loss_ema + 0.1 * total_loss
                total_loss = 0.
                pbar.set_description(f"epoch {epoch + 1} - loss: {loss_ema:.4f}")

            x = x.to(device)
            with torch.no_grad():
                x = ae.encode(x)
            # print(x.min(), x.max())
            loss = ddpm(x)
            loss = loss / accumulate_steps
            total_loss += loss.item()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)

        ddpm.eval()
        with torch.no_grad():
            #reco = ddpm.sample(16, (3, 32, 32), device)
            reco = ddpm.sample(16)
            # print(reco.min(), reco.max())
            grid = make_grid(unnormalize(reco), nrow=4)
            save_image(grid, f"./contents/ddpm_hidden_flowers_{epoch}.png")

            reco = ae.decode(reco)
            reco = unnormalize(reco)
            grid = make_grid(reco, nrow=4)
            save_image(grid, f"./contents/ddpm_flowers_{epoch}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_flowers.pth")