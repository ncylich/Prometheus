import datetime
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# -------------------------------
# Global Hyperparameters
# -------------------------------
epochs = 100
window_size = 60
test_size = 0.2
batch_size = 512
lr = 1e-3

timesteps = 100  # total diffusion steps
beta_start = 1e-4
beta_end = 0.02
hidden_dim = 128      # condition embedding dimension
base_channels = 32   # U-Net base channels

# -------------------------------
# Set Random Seed for Reproducibility
# -------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# -------------------------------
# Diffusion Process Setup Functions
# -------------------------------
def linear_beta_schedule(timesteps, beta_initial=1e-4, beta_final=0.02):
    """Creates a linear schedule for the beta values used in the diffusion process."""
    return torch.linspace(beta_initial, beta_final, timesteps)

def extract(a, t, x_shape):
    """Extracts coefficients for a given timestep."""
    batch_size = t.shape[0]
    out = a.gather(0, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out

# -------------------------------
# Dataset Definition for Multivariate Time Series
# -------------------------------
class MultiStockDataset(Dataset):
    def __init__(self, df, feature_cols, window_size):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        # Create multivariate time series input and corresponding target.
        condition = self.df.loc[idx: idx + self.window_size - 1, self.feature_cols].values.astype(np.float32)
        target = self.df.loc[idx + self.window_size, self.feature_cols].values.astype(np.float32)
        return torch.tensor(condition), torch.tensor(target)

# -------------------------------
# Model Architecture: Bigger U-Net with Skip Connections
# -------------------------------

# (A) Condition Embedding
class ConditionEmbedding(nn.Module):
    def __init__(self, window_size, num_features, hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.SiLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
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
    def __init__(self, in_channels, out_channels, cond_dim, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
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
    def __init__(self, in_channels, out_channels, cond_dim, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.skip_conv = nn.Conv1d(skip_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.SiLU()
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


class DenseBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_dim, cond_dim, signal_length):
        super().__init__()
        flattened_size = in_channels * signal_length
        self.signal_length = signal_length
        self.in_channels = in_channels

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
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

# (D) UNet1D_Large: The larger U-Net.
# This design includes an extra downsampling stage in the encoder (and bottleneck)
# and a corresponding extra upsampling in the decoder.
class UNet1D_Large(nn.Module):
    def __init__(self, in_channels, base_channels, cond_dim, signal_length):
        super().__init__()
        # Calculate signal length at bottleneck
        bottleneck_length = signal_length // 8  # After 3 downsamples

        self.enc1 = ConvBlock(in_channels, base_channels, cond_dim, downsample=False)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, cond_dim, downsample=True)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, cond_dim, downsample=True)
        self.bottleneck = nn.Sequential(
            ConvBlock(base_channels * 4, base_channels * 8, cond_dim, downsample=True),
            DenseBottleneck(base_channels * 8, base_channels * 16, cond_dim, bottleneck_length)
        )

        self.dec1 = UpBlock(base_channels * 8, base_channels * 4, cond_dim, base_channels * 4)
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2, cond_dim, base_channels * 2)
        self.dec3 = UpBlock(base_channels * 2, base_channels, cond_dim, base_channels)
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

# (E) Overall Diffusion Model using the larger U-Net.
class DiffusionTimeSeriesModelUNetLarge(nn.Module):
    def __init__(self, window_size, num_features, hidden_dim=64, base_channels=32):
        super().__init__()
        self.cond_embed = ConditionEmbedding(window_size, num_features, hidden_dim)
        # Reshape the noisy target to [B, 1, num_features] for the U-Net.
        self.unet = UNet1D_Large(in_channels=1, base_channels=base_channels, cond_dim=hidden_dim, signal_length=num_features)
    def forward(self, x, t, condition):
        cond = self.cond_embed(condition, t)  # [B, hidden_dim]
        x_in = x.unsqueeze(1)  # [B, 1, num_features]
        out = self.unet(x_in, cond)  # [B, 1, num_features]
        return out.squeeze(1)

# -------------------------------
# Optimizer, Loss, and Test Functions
# -------------------------------
def calculate_test_loss(model, loader, norm_features, device, timesteps, alphas_bar, loss_fn):
    model.eval()
    total_loss = 0
    close_loss = 0
    volume_loss = 0
    total_samples = 0

    # Identify indices corresponding to close and volume features.
    close_indices = [i for i, feat in enumerate(norm_features) if '_close_norm' in feat]
    volume_indices = [i for i, feat in enumerate(norm_features) if '_volume_norm' in feat]

    with torch.no_grad():
        for condition, target in tqdm(loader):
            condition = condition.to(device)
            target = target.to(device)
            batch_size = target.shape[0]
            t = torch.randint(0, timesteps, (batch_size,), device=device)
            a_bar = extract(alphas_bar, t, target.shape)
            noise = torch.randn_like(target)
            noisy_target = torch.sqrt(a_bar) * target + torch.sqrt(1 - a_bar) * noise
            noise_pred = model(noisy_target, t, condition)
            loss = loss_fn(noise_pred, noise)
            total_loss += loss.item() * batch_size

            close_batch_loss = loss_fn(noise_pred[:, close_indices], noise[:, close_indices])
            volume_batch_loss = loss_fn(noise_pred[:, volume_indices], noise[:, volume_indices])
            close_loss += close_batch_loss.item() * batch_size
            volume_loss += volume_batch_loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_close_loss = close_loss / total_samples
    avg_volume_loss = volume_loss / total_samples
    return avg_loss, avg_close_loss, avg_volume_loss

def validate_one_timestep(model_unet, test_loader, device, timesteps, alphas_bar):
    model_unet.eval()
    with torch.no_grad():
        condition, target = next(iter(test_loader))
        condition_sample = condition[0:1].to(device)  # [1, window_size, num_features]
        target_sample = target[0:1].to(device)          # [1, num_features]
        t = torch.tensor([timesteps // 2], device=device)
        a_bar = extract(alphas_bar, t, target_sample.shape)
        noise = torch.randn_like(target_sample)
        noisy_target = torch.sqrt(a_bar) * target_sample + torch.sqrt(1 - a_bar) * noise
        noise_pred = model_unet(noisy_target, t, condition_sample)
        predicted_target = (noisy_target - torch.sqrt(1 - a_bar) * noise_pred) / torch.sqrt(a_bar)
        noisy_np = noisy_target.cpu().numpy().flatten()
        predicted_np = predicted_target.cpu().numpy().flatten()
        target_np = target_sample.cpu().numpy().flatten()

    feature_indices = np.arange(len(target_np))
    bar_width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(feature_indices - bar_width, noisy_np, width=bar_width, label='Noised Input')
    plt.bar(feature_indices, predicted_np, width=bar_width, label='Diffusion Output')
    plt.bar(feature_indices + bar_width, target_np, width=bar_width, label='Ground Truth')
    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Value")
    plt.title("Validation: Noised Input vs. Diffusion Output vs. Ground Truth (1 Timestep)")
    plt.xticks(feature_indices)
    plt.legend()
    plt.tight_layout()
    plt.show()

class NaiveZeroModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
    def forward(self, x, t, condition):
        return torch.zeros_like(x)

def calculate_naive_test_loss(naive_model, test_loader, norm_features, device, timesteps, alphas_bar, loss_fn):
    return calculate_test_loss(naive_model, test_loader, norm_features, device, timesteps, alphas_bar, loss_fn)

# -------------------------------
# Main Function
# -------------------------------
def main():
    # Data Loading and Preprocessing
    path = '../Local_Data/focused_futures_30min/interpolated_all_long_term_combo.parquet'
    if 'google.colab' in sys.modules:
        path = 'drive/MyDrive' + path[2:]
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')

    target_tickers = ['NIY', 'NKD', 'CL', 'BZ', 'MES', 'ZN', 'MNQ', 'US']
    reg_exp = '^(date|' + '|'.join(target_tickers) + ')'
    df = df.filter(regex=reg_exp)

    feature_cols = df.filter(regex='(_close|_volume)$').columns.tolist()
    print("Features:", feature_cols)

    df = df.sort_values('date').reset_index(drop=True)

    # Normalize features using z-score normalization (after pct_change for close features)
    col_norm_factors = {}
    for col in feature_cols:
        df[col + '_norm'] = df[col].pct_change().fillna(0) if col.endswith('_close') else df[col]
        mean_val = df[col + '_norm'].mean()
        std_val = df[col + '_norm'].std()
        df[col + '_norm'] = (df[col + '_norm'] - mean_val) / std_val
        col_norm_factors[col] = (mean_val, std_val)

    norm_features = [col + '_norm' for col in feature_cols]

    dataset = MultiStockDataset(df, norm_features, window_size)
    train_size_val = int(len(dataset) * (1 - test_size))
    test_size_val = len(dataset) - train_size_val
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size_val, test_size_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Diffusion process parameters
    betas = linear_beta_schedule(timesteps, beta_start, beta_end)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    # Model, Optimizer, and Loss Setup
    num_features = len(norm_features)
    model_unet = DiffusionTimeSeriesModelUNetLarge(window_size, num_features, hidden_dim, base_channels)
    optimizer = optim.Adam(model_unet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,  # Total number of epochs
        eta_min=1e-6,  # Minimum learning rate
    )
    loss_fn = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_unet.to(device)
    betas = betas.to(device)
    alphas_bar = alphas_bar.to(device)

    # Evaluate naive baseline
    naive_model = NaiveZeroModel(num_features).to(device)
    total_loss, c_loss, v_loss = calculate_naive_test_loss(naive_model, test_loader, norm_features, device, timesteps, alphas_bar, loss_fn)
    print(f"Naive baseline - Total loss: {total_loss:.4f}, Close loss: {c_loss:.4f}, Volume loss: {v_loss:.4f}")
    init_unet_loss = calculate_test_loss(model_unet, test_loader, norm_features, device, timesteps, alphas_bar, loss_fn)[0]
    print(f"Initial UNet test loss: {init_unet_loss:.4f}")

    date_string = datetime.datetime.now().strftime('%m-%d-%Y')
    file = f'unet_diffusion_large_best_{date_string}.pth'

    best_c_loss = float('inf')
    corresponding_v_loss = float('inf')

    # Training Loop
    for epoch in range(epochs):
        model_unet.train()
        epoch_loss = 0.0
        for condition, target in tqdm(train_loader):
            condition = condition.to(device)
            target = target.to(device)
            batch_size_current = target.shape[0]
            t = torch.randint(0, timesteps, (batch_size_current,), device=device)
            a_bar = extract(alphas_bar, t, target.shape)
            noise = torch.randn_like(target)
            noisy_target = torch.sqrt(a_bar) * target + torch.sqrt(1 - a_bar) * noise

            noise_pred = model_unet(noisy_target, t, condition)
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * batch_size_current

        epoch_loss /= len(train_dataset)
        total_loss, c_loss, v_loss = calculate_test_loss(model_unet, test_loader, norm_features, device, timesteps, alphas_bar, loss_fn)
        # validate_one_timestep(model_unet, test_loader, device, timesteps, alphas_bar)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Test Loss: Total={total_loss:.4f}, Close={c_loss:.4f}, Volume={v_loss:.4f}")

        if c_loss < best_c_loss:
            best_c_loss = c_loss
            corresponding_v_loss = v_loss
            torch.save(model_unet.state_dict(), file)

    print("Training complete!")
    print("Best Close Loss: {:.4f}, Corresponding Volume: {:.4f}".format(best_c_loss, corresponding_v_loss))

if __name__ == '__main__':
    main()