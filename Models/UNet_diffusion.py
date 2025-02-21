# By end of tonight we will solve quant 🙏.
from datetime import datetime
import pandas as pd
import numpy as np
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
epochs = 25
window_size = 60
test_size = 0.2
batch_size = 1024
lr = 1e-3

timesteps = 100  # total diffusion steps
beta_start = 1e-4
beta_end = 0.02
hidden_dim = 64      # condition embedding dimension
base_channels = 32   # U-Net base channels

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
        # STEP 3: Create dataset for multivariate time series.
        condition = self.df.loc[idx: idx + self.window_size - 1, self.feature_cols].values.astype(np.float32)
        target = self.df.loc[idx + self.window_size, self.feature_cols].values.astype(np.float32)
        return torch.tensor(condition), torch.tensor(target)

# -------------------------------
# Model Architecture: U-Net with Skip Connections
# -------------------------------

# (A) Condition Embedding
class ConditionEmbedding(nn.Module):
    def __init__(self, window_size, num_features, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * num_features, hidden_dim),
            nn.ReLU()
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    def forward(self, condition, t):
        cond_emb = self.fc(condition)
        t = t.float().unsqueeze(1)
        t_emb = self.time_embed(t)
        return cond_emb + t_emb

# (B) ConvBlock for downsampling
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
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

# (C) UpBlock for upsampling
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, skip_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.skip_conv = nn.Conv1d(skip_channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.condition_proj = nn.Linear(cond_dim, out_channels)
    def forward(self, x, skip, cond):
        x = self.up(x)
        # Ensure skip has the same spatial dimension as x.
        if skip.shape[2] != x.shape[2]:
            skip = nn.functional.interpolate(skip, size=x.shape[2], mode='linear', align_corners=False)
        skip_proj = self.skip_conv(skip)
        x = torch.cat([x, skip_proj], dim=1)
        x = self.conv(x)
        cond_proj = self.condition_proj(cond).unsqueeze(2)
        cond_proj = cond_proj.expand(-1, -1, x.shape[2])
        x = x + cond_proj
        return x

# (D) FusionBlock for final decoder stage
class FusionBlock(nn.Module):
    def __init__(self, in_channels, cond_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.condition_proj = nn.Linear(cond_dim, in_channels)
    def forward(self, x, skip, cond):
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        cond_proj = self.condition_proj(cond).unsqueeze(2)
        cond_proj = cond_proj.expand(-1, -1, x.shape[2])
        x = x + cond_proj
        return x

# (E) UNet1D: The overall U-Net.
class UNet1D(nn.Module):
    def __init__(self, in_channels, base_channels, cond_dim, signal_length):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels, cond_dim, downsample=False)
        self.enc2 = ConvBlock(base_channels, base_channels * 2, cond_dim, downsample=True)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 2, cond_dim, downsample=False)
        # In dec1, we use a skip connection from enc2.
        self.dec1 = UpBlock(in_channels=base_channels * 2, out_channels=base_channels,
                            cond_dim=cond_dim, skip_channels=base_channels * 2)
        # In dec2, we use a skip connection from enc1.
        self.dec2 = FusionBlock(in_channels=base_channels, cond_dim=cond_dim)
        self.final_conv = nn.Conv1d(base_channels, in_channels, kernel_size=3, padding=1)
    def forward(self, x, cond):
        e1 = self.enc1(x, cond)      # [B, base_channels, L]
        e2 = self.enc2(e1, cond)     # [B, base_channels*2, L//2]
        b = self.bottleneck(e2, cond)  # [B, base_channels*2, L//2]
        # Upsample e2 to match the spatial dimensions for the skip connection.
        e2_up = nn.functional.interpolate(e2, scale_factor=2, mode='linear', align_corners=False)
        d1 = self.dec1(b, e2_up, cond)  # [B, base_channels, L]
        d2 = self.dec2(d1, e1, cond)     # [B, base_channels, L]
        out = self.final_conv(d2)        # [B, in_channels, L]
        return out

# (F) Overall Diffusion Model using U-Net.
class DiffusionTimeSeriesModelUNet(nn.Module):
    def __init__(self, window_size, num_features, hidden_dim=64, base_channels=32):
        super().__init__()
        self.cond_embed = ConditionEmbedding(window_size, num_features, hidden_dim)
        # The noisy target vector is reshaped to [B, 1, num_features]
        self.unet = UNet1D(in_channels=1, base_channels=base_channels, cond_dim=hidden_dim, signal_length=num_features)
    def forward(self, x, t, condition):
        cond = self.cond_embed(condition, t)  # [B, hidden_dim]
        x_in = x.unsqueeze(1)  # [B, 1, num_features]
        out = self.unet(x_in, cond)  # [B, 1, num_features]
        return out.squeeze(1)

# -------------------------------
# Optimizer, Loss, and Test Functions
# -------------------------------
def calculate_test_loss(model, loader, norm_features, device, timesteps, alphas_bar, loss_fn):
    """
    Calculates the test loss along with separate losses for close and volume features.
    """
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
        # Get one batch from the test loader and pick the first sample.
        condition, target = next(iter(test_loader))
        condition_sample = condition[0:1].to(device)  # shape: [1, window_size, num_features]
        target_sample = target[0:1].to(device)  # shape: [1, num_features]

        # Choose a fixed diffusion timestep for clarity, e.g., t = timesteps // 2.
        t = torch.tensor([timesteps // 2], device=device)
        a_bar = extract(alphas_bar, t, target_sample.shape)

        # Forward process: add noise to create the noised target.
        noise = torch.randn_like(target_sample)
        noisy_target = torch.sqrt(a_bar) * target_sample + torch.sqrt(1 - a_bar) * noise

        # Model prediction: predict the noise given the noised target and condition.
        noise_pred = model_unet(noisy_target, t, condition_sample)

        # Reverse process: compute the predicted (denoised) target.
        predicted_target = (noisy_target - torch.sqrt(1 - a_bar) * noise_pred) / torch.sqrt(a_bar)

        # Convert tensors to numpy arrays for plotting.
        noisy_np = noisy_target.cpu().numpy().flatten()
        predicted_np = predicted_target.cpu().numpy().flatten()
        target_np = target_sample.cpu().numpy().flatten()

    # Create a grouped bar chart over the feature indices.
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
    # -------------------------------
    # Data Loading and Preprocessing
    # -------------------------------
    path = '../Local_Data/focused_futures_30min/interpolated_all_long_term_combo.parquet'
    if 'google.colab' in sys.modules:
        path = 'drive/MyDrive' + path[2:]
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')

    # STEP 1: Select Close and Volume features (for chosen tickers)
    target_tickers = ['NIY', 'NKD', 'CL', 'BZ', 'MES', 'ZN', 'MNQ', 'US']
    reg_exp = '^(date|' + '|'.join(target_tickers) + ')'
    df = df.filter(regex=reg_exp)

    # Extract all columns ending with '_close' or '_volume'
    feature_cols = df.filter(regex='(_close|_volume)$').columns.tolist()
    print("Features:", feature_cols)

    df = df.sort_values('date').reset_index(drop=True)

    # STEP 2: Normalize each feature using z-score normalization.
    col_norm_factors = {}
    for col in feature_cols:
        df[col + '_norm'] = df[col].pct_change().fillna(0) if col.endswith('_close') else df[col]
        mean_val = df[col + '_norm'].mean()
        std_val = df[col + '_norm'].std()
        df[col + '_norm'] = (df[col + '_norm'] - mean_val) / std_val
        col_norm_factors[col] = (mean_val, std_val)

    norm_features = [col + '_norm' for col in feature_cols]

    # Create dataset for multivariate time series.
    dataset = MultiStockDataset(df, norm_features, window_size)
    train_size_val = int(len(dataset) * (1 - test_size))
    test_size_val = len(dataset) - train_size_val
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size_val, test_size_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------
    # Diffusion Process Setup
    # -------------------------------
    betas = linear_beta_schedule(timesteps, beta_start, beta_end)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    # -------------------------------
    # Model, Optimizer, Loss, and Device Setup
    # -------------------------------
    num_features = len(norm_features)
    model_unet = DiffusionTimeSeriesModelUNet(window_size, num_features, hidden_dim, base_channels)
    optimizer = optim.Adam(model_unet.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_unet.to(device)
    betas = betas.to(device)
    alphas_bar = alphas_bar.to(device)

    # Naive model baseline
    naive_model = NaiveZeroModel(num_features).to(device)
    total_loss, c_loss, v_loss = calculate_naive_test_loss(naive_model, test_loader, norm_features, device, timesteps, alphas_bar, loss_fn)
    print(f"Naive baseline - Total loss: {total_loss:.4f}, Close loss: {c_loss:.4f}, Volume loss: {v_loss:.4f}")
    init_unet_loss = calculate_test_loss(model_unet, test_loader, norm_features, device, timesteps, alphas_bar, loss_fn)[0]
    print(f"Initial UNet test loss: {init_unet_loss:.4f}")

    # -------------------------------
    # Training Loop
    # -------------------------------
    date_string = datetime.now().strftime('%m-%d-%Y')
    file = f'unet_diffusion_{date_string}.pth'

    best_c_loss = float('inf')
    corresponding_v_loss = float('inf')

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

            epoch_loss += loss.item() * batch_size_current

        epoch_loss /= len(train_dataset)
        total_loss, c_loss, v_loss = calculate_test_loss(model_unet, test_loader, norm_features, device, timesteps, alphas_bar, loss_fn)
        validate_one_timestep(model_unet, test_loader, device, timesteps, alphas_bar)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Test Loss: Total={total_loss:.4f}, Close={c_loss:.4f}, Volume={v_loss:.4f}")

        if c_loss < best_c_loss:
            best_c_loss = c_loss
            corresponding_v_loss = v_loss
            torch.save(model_unet.state_dict(), file)

    print("Training complete!")
    print("Best Close Loss: {:.4f}, Corresponding Volume: {:.4f}".format(best_c_loss, corresponding_v_loss))

# Only run the main function if this script is executed directly
if __name__ == '__main__':
    main()