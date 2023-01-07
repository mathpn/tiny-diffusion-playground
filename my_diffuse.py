"""
Draft implementation of Difussion model.
"""

# pylint: disable=no-member
import os
import torch
from torch import nn
from torchvision import transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        out = self.conv2(x1)
        if self.same_channels:
            out = out + x
        else:
            out = out + x1
        return out


class DownConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ResidualBlock(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        return self.downsample(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv2 = ResidualBlock(out_channels, out_channels)
        self.conv3 = ResidualBlock(out_channels, out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class UNet(nn.Module):
    # https://arxiv.org/abs/2006.11239
    def __init__(self, in_channels: int, out_channels: int, n_feat: int):
        super().__init__()
        self.conv1 = ResidualBlock(in_channels, n_feat)
        self.down1 = DownConv(n_feat, n_feat)
        self.down2 = DownConv(n_feat, 2 * n_feat)
        # self.down3 = DownConv(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.LeakyReLU())

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.BatchNorm2d(2 * n_feat),
            nn.ReLU(),
        )
        # self.up1 = UpConv(4 * n_feat, 2 * n_feat)
        self.up2 = UpConv(4 * n_feat, n_feat)
        self.up3 = UpConv(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, out_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        # d3 = self.down3(d2)

        # thro = self.to_vec(d3)
        z = self.to_vec(d2)
        # TODO time embedding

        thro = self.up0(z)
        # up1 = self.up1(thro, d3)
        # up2 = self.up2(up1, d2)
        up2 = self.up2(thro, d2)
        up3 = self.up3(up2, d1)

        out = self.out(torch.cat((up3, x), 1))
        return out, z


class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ):
        super().__init__()
        self.eps_model = eps_model
        self.criterion = criterion
        self.n_T = n_T

        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ts = torch.randint(1, self.n_T + 1, x.shape[0:1]).to(x.device)
        eps = torch.randn_like(x)

        x_t = (
            self.sqrtab[ts, None, None, None] * x + self.sqrtmab[ts, None, None, None] * eps
            # self.sqrtab[ts, None] * x + self.sqrtmab[ts, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        eps_h, z = self.eps_model(x_t, ts / self.n_T)
        return self.criterion(eps, eps_h), z

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps, _ = self.eps_model(x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1))
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

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


def kl_div(z):
    var = z.var()
    logvar = var.log()
    mu = z.mean()
    return 0.5 * torch.mean(torch.pow(mu, 2) + var - 1.0 - logvar)


def train_mnist():
    os.makedirs("./contents", exist_ok=True)

    n_epochs = 40
    betas = (1e-4, 0.02)
    n_T = 500
    n_feat = 128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    unet = UNet(1, 1, n_feat)
    ddpm = DDPM(unet, betas, n_T)
    ddpm.to(device)

    print(sum(p.numel() for p in ddpm.parameters()))
    transform = T.Compose([T.ToTensor()])
    dataset = MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=10)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    for epoch in range(n_epochs):
        ddpm.train()

        pbar = tqdm(loader)
        loss_ema = None
        kl_ema = None
        for x, _ in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss, z = ddpm(x)
            loss_kl = kl_div(z)
            loss = loss + 0.05 * loss_kl
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
                kl_ema = loss_kl.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
                kl_ema = 0.9 * kl_ema + 0.1 * loss_kl.item()
            pbar.set_description(f"loss: {loss_ema:.4f} KL: {kl_ema:.2f}")
            optim.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(16, (1, 28, 28), device)
            grid = make_grid(xh, nrow=4)
            save_image(grid, f"./contents/ddpm_sample_{epoch}.png")

            # save model
            torch.save(ddpm.state_dict(), f"./ddpm_mnist.pth")


if __name__ == "__main__":
    # unet = UNet(3, 3, 32)
    # out = unet(torch.randn(1, 3, 28, 28), 0)
    # print(out.shape)

    # ddpm = DDPM(unet, (0.1, 0.9), 10)
    # loss = ddpm(torch.randn(5, 3, 28, 28))
    # print(loss)
    train_mnist()
