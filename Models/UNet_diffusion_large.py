# model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import datetime

# -------------------------------
# Global Hyperparameters
# -------------------------------
EPOCHS = 100
WINDOW_SIZE = 64
TEST_SIZE = 0.2
BATCH_SIZE = 512
LR = 1e-3
DROPOUT_RATE = 0.2

L1_WEIGHT = 1e-5
L2_WEIGHT = 1e-5
TIMESTEPS = 100
BETA_START = 1e-4
BETA_END = 0.1
HIDDEN_DIM = 128
BASE_CHANNELS = 32

# -------------------------------
# Set Random Seed for Reproducibility
# -------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# -------------------------------
# Model Architecture
# -------------------------------

# (A) Condition Embedding remains unchanged.
class ConditionEmbedding(nn.Module):
    def __init__(self, window_size, num_features, hidden_dim, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(1)
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, condition, t):
        # Transpose condition from [B, W, F] to [B, F, W]
        condition = condition.transpose(1, 2)
        cond_emb = self.cnn(condition).squeeze(-1)  # [B, hidden_dim]
        t = t.float().unsqueeze(1)
        t_emb = self.time_embed(t)
        return cond_emb + t_emb

# (B) ConvBlock for Downsampling
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, downsample=False, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.downsample = downsample
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        self.condition_proj = nn.Linear(cond_dim, out_channels)
        if downsample:
            self.pool = nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, cond):
        h = self.conv(x)
        cond_proj = self.condition_proj(cond).unsqueeze(2)
        h = h + cond_proj
        if self.downsample:
            h = self.pool(h)
        return h

# (C) UpBlock for Upsampling
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, skip_channels, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.skip_conv = nn.Conv1d(skip_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        self.condition_proj = nn.Linear(cond_dim, out_channels)

    def forward(self, x, skip, cond):
        x = self.up(x)
        if skip.shape[2] != x.shape[2]:
            skip = nn.functional.interpolate(skip, size=x.shape[2], mode='linear', align_corners=False)
        skip_proj = self.skip_conv(skip)
        x = torch.cat([x, skip_proj], dim=1)
        x = self.conv(x)
        cond_proj = self.condition_proj(cond).unsqueeze(2)
        cond_proj = cond_proj.expand(-1, -1, x.shape[2])
        x = x + cond_proj
        return x

# (D) DenseBottleneck block remains unchanged.
class DenseBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_dim, cond_dim, signal_length, dropout_rate=DROPOUT_RATE):
        super().__init__()
        flattened_size = in_channels * signal_length
        self.signal_length = signal_length
        self.in_channels = in_channels

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, flattened_size)
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, flattened_size)
        )

    def forward(self, x, cond):
        b = self.dense(x)
        cond_proj = self.cond_proj(cond)
        b = b + cond_proj
        return b.view(-1, self.in_channels, self.signal_length)

# (E) UNet1D_Large: The larger U-Net.
class UNet1D_Large(nn.Module):
    def __init__(self, in_channels, base_channels, cond_dim, signal_length, dropout_rate=DROPOUT_RATE):
        super().__init__()
        bottleneck_length = signal_length // 8

        self.enc1 = ConvBlock(in_channels, base_channels, cond_dim, downsample=False, dropout_rate=dropout_rate)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, cond_dim, downsample=True, dropout_rate=dropout_rate)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, cond_dim, downsample=True, dropout_rate=dropout_rate)
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, cond_dim, downsample=True, dropout_rate=dropout_rate),
            DenseBottleneck(base_channels * 8, base_channels * 16, cond_dim, bottleneck_length, dropout_rate=dropout_rate)
        )

        self.dec1 = UpBlock(base_channels * 8, base_channels * 4, cond_dim, base_channels * 4, dropout_rate=dropout_rate)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2, cond_dim, base_channels * 2, dropout_rate=dropout_rate)
        self.dec3 = UpBlock(base_channels * 2, base_channels, cond_dim, base_channels, dropout_rate=dropout_rate)
        self.final_conv = nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, cond):
        e1 = self.enc1(x, cond)
        e2 = self.enc2(e1, cond)
        e3 = self.enc3(e2, cond)
        b = self.bottleneck[0](e3, cond)
        b = self.bottleneck[1](b, cond)
        d1 = self.dec1(b, e3, cond)
        d2 = self.dec2(d1, e2, cond)
        d3 = self.dec3(d2, e1, cond)
        return self.final_conv(d3)

# (F) Overall Diffusion Model using the larger U-Net.
# This model accepts the full condition (price & volume) but outputs only price predictions.
class DiffusionTimeSeriesModelUNetLarge(nn.Module):
    def __init__(self, window_size, input_features, output_features, hidden_dim=HIDDEN_DIM, base_channels=BASE_CHANNELS):
        super().__init__()
        self.cond_embed = ConditionEmbedding(window_size, input_features, hidden_dim, dropout_rate=DROPOUT_RATE)
        self.unet = UNet1D_Large(in_channels=1, base_channels=base_channels, cond_dim=hidden_dim,
                                 signal_length=output_features, dropout_rate=DROPOUT_RATE)

    def forward(self, x, t, condition):
        # x: noisy price target [B, output_features]
        cond = self.cond_embed(condition, t)  # [B, hidden_dim]
        x_in = x.unsqueeze(1)                # [B, 1, output_features]
        out = self.unet(x_in, cond)          # [B, 1, output_features]
        return out.squeeze(1)


# -------------------------------
# Main Method: Call Training & Data Processing Code
# -------------------------------
def main():
    # Import the training classes and methods from the training file
    from Train.train_unet_diff import DiffusionTrainer

    # Instantiate the trainer with our hyperparameters and the model class
    trainer = DiffusionTrainer(
        epochs=EPOCHS,
        window_size=WINDOW_SIZE,
        test_size=TEST_SIZE,
        batch_size=BATCH_SIZE,
        lr=LR,
        l1_weight=L1_WEIGHT,
        l2_weight=L2_WEIGHT,
        timesteps=TIMESTEPS,
        beta_start=BETA_START,
        beta_end=BETA_END,
        hidden_dim=HIDDEN_DIM,
        base_channels=BASE_CHANNELS,
        dropout_rate=DROPOUT_RATE,
        model_class=DiffusionTimeSeriesModelUNetLarge  # Pass our model architecture
    )
    # Run training (this will load data, preprocess, train, and evaluate)
    trainer.run_training()

if __name__ == '__main__':
    main()
