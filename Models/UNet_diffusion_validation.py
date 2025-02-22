#!/usr/bin/env python
import argparse
import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from UNet_diffusion import (
    DiffusionTimeSeriesModelUNet,
    linear_beta_schedule,
    extract,
    batch_size
)

# -------------------------------
# Set Random Seed for Reproducibility
# -------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# -------------------------------
# Full Reverse Diffusion Inference Function
# -------------------------------
def full_inference(model, condition, betas, alphas, alphas_bar, timesteps, device):
    """
    Runs full reverse diffusion starting from pure noise.
    Inputs:
      - model: the diffusion model.
      - condition: historical window tensor of shape [B, window_size, num_features].
      - betas, alphas, alphas_bar: diffusion process tensors.
      - timesteps: total number of diffusion steps.
      - device: torch device.
    Returns:
      - x: final denoised sample tensor of shape [B, num_features].
    """
    model.eval()
    batch_size, num_features = condition.shape[0], condition.shape[2]
    x = torch.randn(batch_size, num_features, device=device)

    with torch.no_grad():
        for t in tqdm(reversed(range(1, timesteps))):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            a_bar_t = extract(alphas_bar, t_tensor, x.shape)
            noise_pred = model(x, t_tensor, condition)
            x0_pred = (x - torch.sqrt(1 - a_bar_t) * noise_pred) / torch.sqrt(a_bar_t)

            t_prev = t - 1
            t_prev_tensor = torch.full((batch_size,), t_prev, device=device, dtype=torch.long)
            a_bar_prev = extract(alphas_bar, t_prev_tensor, x.shape)

            noise = torch.randn_like(x) if t_prev > 0 else 0.0
            x = torch.sqrt(a_bar_prev) * x0_pred + torch.sqrt(1 - a_bar_prev) * noise

        # Final step at t=0
        t_tensor = torch.zeros(batch_size, device=device, dtype=torch.long)
        a_bar_0 = extract(alphas_bar, t_tensor, x.shape)
        noise_pred = model(x, t_tensor, condition)
        x = (x - torch.sqrt(1 - a_bar_0) * noise_pred) / torch.sqrt(a_bar_0)

    return x

# -------------------------------
# Dataset for Multivariate Time Series
# -------------------------------
class MultiStockDataset(Dataset):
    def __init__(self, df, feature_cols, window_size):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        # Condition: historical window (window_size timesteps) for all features
        condition = self.df.loc[idx: idx + self.window_size - 1, self.feature_cols].values.astype(np.float32)
        # Target: next timestep's vector (all features)
        target = self.df.loc[idx + self.window_size, self.feature_cols].values.astype(np.float32)
        return torch.tensor(condition), torch.tensor(target)

# -------------------------------
# Naive Zero Model for MSE Calculation
# -------------------------------
class NaiveZeroModel:
    def __init__(self, device):
        self.device = device

    def __call__(self, x, t, condition):
        # Create zeros tensor with same shape as target (last timestep, all features)
        return torch.zeros(condition.shape[0], condition.shape[2], device=self.device)

def calculate_naive_mse(test_loader, device, close_indices, volume_indices):
    """
    Calculate MSE for a naive model that always predicts zeros, separately for close and volume features.
    """
    naive_model = NaiveZeroModel(device)
    mse_loss_fn = torch.nn.MSELoss(reduction='sum')
    total_overall = 0.0
    total_close = 0.0
    total_volume = 0.0
    total_samples = 0

    with torch.no_grad():
        for condition, target in tqdm(test_loader, desc='Naive Zero Model'):
            condition = condition.to(device)
            target = target.to(device)
            pred = naive_model(None, None, condition)
            overall_loss = mse_loss_fn(pred, target)
            close_loss = mse_loss_fn(pred[:, close_indices], target[:, close_indices])
            volume_loss = mse_loss_fn(pred[:, volume_indices], target[:, volume_indices])
            total_overall += overall_loss.item()
            total_close += close_loss.item()
            total_volume += volume_loss.item()
            total_samples += target.shape[0]

    overall_mse = total_overall / total_samples
    close_mse = total_close / total_samples
    volume_mse = total_volume / total_samples
    return overall_mse, close_mse, volume_mse

# -------------------------------
# Main Function
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full Inference and MSE Evaluation for Diffusion Time Series Model")
    parser.add_argument("--model_path", type=str, default='unet_diffusion_best.pth', help="Path to the saved model")
    parser.add_argument("--window_size", type=int, default=60, help="Window size")
    parser.add_argument("--num_features", type=int, default=16, help="Number of features")
    parser.add_argument("--timesteps", type=int, default=100, help="Diffusion steps")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--base_channels", type=int, default=32, help="Base channels")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up diffusion process tensors
    betas = linear_beta_schedule(args.timesteps, beta_initial=1e-4, beta_final=0.02).to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0).to(device)

    # Load the saved model and move it to device
    model = DiffusionTimeSeriesModelUNet(args.window_size, args.num_features,
                                         args.hidden_dim, args.base_channels)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # -------------------------------
    # Load Test Set
    # -------------------------------
    # Load futures data; the dataframe should contain columns like 'date', 'CL_close', 'CL_volume', etc.
    df = pd.read_parquet('../Local_Data/focused_futures_30min/interpolated_all_long_term_combo.parquet')
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')

    # STEP 1: Select Close and Volume features (for multiple instruments)
    target_tickers = ['NIY', 'NKD', 'CL', 'BZ', 'MES', 'ZN', 'MNQ', 'US']
    reg_exp = '^(date|' + '|'.join(target_tickers) + ')'
    df = df.filter(regex=reg_exp)

    # Extract all columns that end with '_close' or '_volume'
    feature_cols = df.filter(regex='(_close|_volume)$').columns.tolist()
    print("Features:", feature_cols)

    # Sort by date and reset index
    df = df.sort_values('date').reset_index(drop=True)

    # STEP 2: Normalize each feature using z-score normalization.
    # For '_close' columns, we use pct_change; for volume, we use the raw value.
    col_norm_factors = {}
    for col in feature_cols:
        df[col + '_norm'] = df[col].pct_change().fillna(0) if col.endswith('_close') else df[col]
        mean_val = df[col + '_norm'].mean()
        std_val = df[col + '_norm'].std()
        df[col + '_norm'] = (df[col + '_norm'] - mean_val) / std_val
        col_norm_factors[col] = (mean_val, std_val)

    # List of normalized feature columns
    norm_features = [col + '_norm' for col in feature_cols]

    # Determine indices for close and volume features based on the normalized column names
    close_indices = [i for i, f in enumerate(norm_features) if '_close' in f]
    volume_indices = [i for i, f in enumerate(norm_features) if '_volume' in f]

    # Create test dataset and dataloader
    dataset = MultiStockDataset(df, norm_features, args.window_size)
    test_size_val = int(len(dataset) * 0.2)
    _, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - test_size_val, test_size_val])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------
    # Run Full Inference on the Test Set and Calculate MSE
    # -------------------------------
    # Calculate baseline MSE with naive zero model separately for close and volume
    naive_overall, naive_close, naive_volume = calculate_naive_mse(test_loader, device, close_indices, volume_indices)
    print(f"Naive Zero Model MSE on test set: overall {naive_overall:.6f}, close {naive_close:.6f}, volume {naive_volume:.6f}")

    mse_loss_fn = torch.nn.MSELoss(reduction='sum')
    total_overall = 0.0
    total_close = 0.0
    total_volume = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for condition, target in tqdm(test_loader):
            condition = condition.to(device)  # shape: [B, window_size, num_features]
            target = target.to(device)          # shape: [B, num_features]
            # Run full reverse diffusion to generate prediction from the condition
            pred = full_inference(model, condition, betas, alphas, alphas_bar, args.timesteps, device)
            # Compute MSE between the predicted target and the actual target for overall, close, and volume features
            overall_loss = mse_loss_fn(pred, target)
            close_loss = mse_loss_fn(pred[:, close_indices], target[:, close_indices])
            volume_loss = mse_loss_fn(pred[:, volume_indices], target[:, volume_indices])
            total_overall += overall_loss.item()
            total_close += close_loss.item()
            total_volume += volume_loss.item()
            total_samples += target.shape[0]

    overall_mse = total_overall / total_samples
    close_mse = total_close / total_samples
    volume_mse = total_volume / total_samples
    print(f"Diffusion Model MSE on test set: overall {overall_mse:.6f}, close {close_mse:.6f}, volume {volume_mse:.6f}")

if __name__ == '__main__':
    main()