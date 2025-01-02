import sys

if 'google.colab' in sys.modules:
    from Prometheus.Train.train_vae import train_model
    from Prometheus.Train.dataloaders import get_long_term_Xmin_data_loaders
    from Prometheus.Models.load_config import dynamic_load_config, update_config_with_factor
else:
    from Train.train_vae import train_model
    from Train.dataloaders import get_long_term_Xmin_data_loaders
    from Models.load_config import dynamic_load_config, update_config_with_factor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.nn import MSELoss
import torch.nn.init as init
import math
from dataclasses import dataclass


@dataclass
class Config:
    sequence_len: int = 16

    latent_dim: int = 16
    use_dct: bool = False
    num_tickers: int = 8
    embed_dim: int = 128

    ff_dim: int = 256
    batch_size: int = 1024
    lr: float = 1e-3
    epochs: int = 100
    init_weight_magnitude: float = 1e-3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StockVAE(nn.Module):
    def __init__(self, num_tickers=8, sequence_len=30, latent_dim=16, use_dct=False):
        super(StockVAE, self).__init__()
        self.num_tickers = num_tickers
        self.sequence_len = sequence_len
        self.latent_dim = latent_dim
        self.use_dct = use_dct

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(num_tickers, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample to sequence_len // 2
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample to sequence_len // 4
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Downsample to sequence_len // 8
        )
        self.fc1 = nn.Linear(128 * (sequence_len // 8), 256)
        self.fc2_mu = nn.Linear(256, latent_dim)
        self.fc2_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 128 * (sequence_len // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsample to sequence_len // 4
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsample to sequence_len // 2
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Upsample to sequence_len
            nn.ConvTranspose1d(16, num_tickers, kernel_size=3, stride=1, padding=1),
        )

        # DCT matrices
        self.dct_matrix, self.idct_matrix = self.get_dct_matrix(sequence_len)

    def get_dct_matrix(self, N):
        """Calculates DCT Matrix of size N."""
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)

        return torch.tensor(dct_m, dtype=torch.float32), torch.tensor(idct_m, dtype=torch.float32)

    def dct_encode(self, x):
        return torch.matmul(x, self.dct_matrix.to(x.device))

    def dct_decode(self, x):
        return torch.matmul(x, self.idct_matrix.to(x.device))

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten
        h = F.relu(self.fc1(h))
        return self.fc2_mu(h), self.fc2_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        h = h.view(h.size(0), 128, self.sequence_len // 8)  # Reshape
        return torch.sigmoid(self.decoder(h))

    def forward(self, x):
        if self.use_dct:
            x = self.dct_encode(x)

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)

        if self.use_dct:
            recon_x = self.dct_decode(recon_x)

        return recon_x, mu, logvar


def main(config_path: str = ''):
    config = dynamic_load_config(config_path, Config)
    config = update_config_with_factor(config)

    train_loader, test_loader = get_long_term_Xmin_data_loaders(config.sequence_len, config.sequence_len,
                                                                config.num_tickers, x_min=30, batch_size=config.batch_size)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight, mean=0, std=config.init_weight_magnitude)
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            init.normal_(m.weight, mean=0, std=config.init_weight_magnitude)

    model = StockVAE(num_tickers=config.num_tickers,
                     sequence_len=config.sequence_len,
                     latent_dim=config.latent_dim,
                     use_dct=config.use_dct).to(device)

    if config.init_weight_magnitude:
        model.apply(init_weights)

    optimizer = AdamW(model.parameters(), lr=config.lr)

    patience = max(1, math.floor(math.log(config.epochs, math.e)))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience)

    def criterion(recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD

    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config.epochs, device)

if __name__ == '__main__':
    main('configs/vae_config.yaml')