import sys

if 'google.colab' in sys.modules:
    from Prometheus.Train.train_vae import train_model
    from Prometheus.Train.dataloaders import get_long_term_Xmin_data_loaders
    from Prometheus.Models.load_config import dynamic_load_config, update_config_with_factor
else:
    from Train.train_vae import train_model
    from Train.dataloaders import get_long_term_Xmin_data_loaders
    from Models.Unsuccessful_Models.load_config import dynamic_load_config, update_config_with_factor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch.nn.init as init
import math
from dataclasses import dataclass

@dataclass
class Config:
    seq_len: int = 16
    in_channels: int = 2

    latent_dim: int = 16
    use_dct: bool = False
    num_tickers: int = 8
    embed_dim: int = 128

    ff_dim: int = 256
    batch_size: int = 1024
    lr: float = 1e-3
    epochs: int = 100
    init_weight_magnitude: float = 1e-3

    kld_skip: int = 0
    kld_beta: float = 0.1
    klm_mp_divisor: int = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockVAE(nn.Module):
    def __init__(
        self,
        num_tickers=8,
        seq_len=16,
        latent_dim=16,
        in_channels=2,
        use_dct=False,
        large_model=True
    ):
        super(StockVAE, self).__init__()
        self.num_tickers = num_tickers
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.use_dct = use_dct
        self.large_model = large_model
        self.in_channels = in_channels

        if large_model:
            # -------------------------
            # Encoder
            # -------------------------
            self.encoder1 = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  # (B, 32, 4, 8)
            )
            self.encoder2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  # (B, 64, 2, 4)
            )
            self.encoder3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),  # (B, 128, 1, 2)
            )

            # Flatten -> FC for (mu, logvar)
            self.fc1 = nn.Linear(128 * (seq_len // 8) * (num_tickers // 8), 128)
            self.fc2_mu = nn.Linear(128, latent_dim)
            self.fc2_logvar = nn.Linear(128, latent_dim)

            # -------------------------
            # Decoder
            # -------------------------
            self.fc3 = nn.Linear(latent_dim, 128)
            self.fc4 = nn.Linear(
                128, 128 * (seq_len // 8) * (num_tickers // 8)
            )  # => shape (B, 256 if 8x16)

            # Decoder “blocks” without skip connections
            # 1) (B, 128, 1, 2) -> upsample -> (B, 128, 2, 4)
            self.decoder1 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
            self.up1 = nn.Upsample(scale_factor=2, mode="nearest")

            # 2) (B, 128, 2, 4) -> upsample -> (B, 128, 4, 8)
            #    No longer concatenating, so we only have 128 in-channels
            self.decoder2 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

            # 3) (B, 64, 4, 8) -> upsample -> (B, 64, 8, 16)
            self.decoder3 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
            )

            self.final_layer = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)

        else:
            # -------------------------
            # Smaller Encoder–Decoder (unchanged)
            # -------------------------
            self.encoder1 = nn.Sequential(
                nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=0),
                nn.ReLU()
            )
            self.encoder2 = nn.Sequential(
                nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0),
                nn.ReLU()
            )
            self.fc1 = nn.Linear(8 * (seq_len - 4) * (num_tickers - 4), 64)
            self.fc2_mu = nn.Linear(64, latent_dim)
            self.fc2_logvar = nn.Linear(64, latent_dim)

            self.fc3 = nn.Linear(latent_dim, 64)
            self.fc4 = nn.Linear(64, 8 * seq_len * num_tickers)
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1, padding=1),
                nn.ReLU()
            )
            self.final_layer = nn.Sequential(
                nn.ConvTranspose2d(4, in_channels, kernel_size=3, stride=1, padding=1),
            )

        # DCT (optional)
        self.dct_matrix, self.idct_matrix = self.get_dct_matrix(seq_len)

    def get_dct_matrix(self, N):
        """Calculates DCT Matrix of size N."""
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 0.5) * k / N)
        idct_m = np.linalg.inv(dct_m)
        return torch.tensor(dct_m, dtype=torch.float32), torch.tensor(idct_m, dtype=torch.float32)

    def dct_encode(self, x):
        # x: (B, C, H, W)
        return torch.matmul(x, self.dct_matrix.to(x.device))

    def dct_decode(self, x):
        return torch.matmul(x, self.idct_matrix.to(x.device))

    def encode(self, x):
        if self.large_model:
            h1 = self.encoder1(x)  # (B, 32, 4, 8)
            h2 = self.encoder2(h1) # (B, 64, 2, 4)
            h3 = self.encoder3(h2) # (B, 128, 1, 2)

            h_flat = h3.view(h3.size(0), -1)
            h_fc = F.relu(self.fc1(h_flat))
            mu = self.fc2_mu(h_fc)
            logvar = self.fc2_logvar(h_fc)
            return mu, logvar, h1, h2, h3

        else:
            h1 = self.encoder1(x)
            h2 = self.encoder2(h1)
            h_flat = h2.view(h2.size(0), -1)
            h_fc = F.relu(self.fc1(h_flat))
            mu = self.fc2_mu(h_fc)
            logvar = self.fc2_logvar(h_fc)
            return mu, logvar, h1, h2, None

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, h1=None, h2=None, h3=None):
        """
        Decode without skip connections (for large_model=True).
        """
        if not self.large_model:
            # Smaller decoder
            h = F.relu(self.fc3(z))
            h = F.relu(self.fc4(h))
            h = h.view(h.size(0), 8, self.num_tickers, self.seq_len)
            h = self.decoder1(h)
            return torch.sigmoid(self.final_layer(h))

        # Large model (no skip connections)
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        # Reshape => (B, 128, 1, 2)
        h = h.view(h.size(0), 128, self.num_tickers // 8, self.seq_len // 8)

        # 1) (B, 128, 1, 2) -> conv -> upsample => (B, 128, 2, 4)
        h = self.decoder1(h)
        h = self.up1(h)

        # 2) (B, 128, 2, 4) -> conv -> upsample => (B, 64, 4, 8)
        h = self.decoder2(h)
        h = self.up2(h)

        # 3) (B, 64, 4, 8) -> conv -> interpolate => (B, 32, 8, 16)
        h = self.decoder3(h)
        h = F.interpolate(h, scale_factor=2, mode="nearest")

        # Final => (B, 2, 8, 16)
        h = self.final_layer(h)
        return h

    def forward(self, x):
        if self.use_dct:
            x = self.dct_encode(x)

        mu, logvar, h1, h2, h3 = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, h1, h2, h3)

        if self.use_dct:
            recon_x = self.dct_decode(recon_x)

        return recon_x, mu, logvar


def main(config_path: str = ''):
    config = dynamic_load_config(config_path, Config)
    config = update_config_with_factor(config)

    train_loader, test_loader = get_long_term_Xmin_data_loaders(
        config.seq_len, config.seq_len,
        config.num_tickers, x_min=30,
        batch_size=config.batch_size
    )

    def init_weights(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0, std=config.init_weight_magnitude)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            init.normal_(m.weight, mean=0, std=config.init_weight_magnitude)

    model = StockVAE(
        num_tickers=config.num_tickers,
        seq_len=config.seq_len,
        latent_dim=config.latent_dim,
        in_channels=config.in_channels,
        use_dct=config.use_dct,
        large_model=True  # will now have NO skip connections in the decoder
    ).to(device)

    if config.init_weight_magnitude:
        model.apply(init_weights)

    optimizer = AdamW(model.parameters(), lr=config.lr)

    patience = max(1, math.floor(math.log(config.epochs, math.e)))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience)

    def criterion(recon_x, x, mu, logvar, kld_weight):
        mse = F.mse_loss(recon_x, x)
        # Standard VAE KL
        kld = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld_weight * kld, mse, kld

    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config.epochs, config, device)

if __name__ == '__main__':
    main('../configs/vae_config.yaml')
