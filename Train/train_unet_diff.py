# train.py
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
import os
import matplotlib.pyplot as plt


# -------------------------------
# Extracts coefficients for a given timestep in the diffusion process.
# -------------------------------
def extract(a, t, x_shape):
    """Extracts coefficients for a given timestep."""
    batch_size = t.shape[0]
    out = a.gather(0, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out


# -------------------------------
# Dataset Definition for Multivariate Time Series
# -------------------------------
class MultiStockDataset(Dataset):
    def __init__(self, df, condition_cols, target_cols, window_size):
        self.df = df.reset_index(drop=True)
        self.condition_cols = condition_cols
        self.target_cols = target_cols
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        # Condition: all features (price and volume)
        condition = self.df.loc[idx: idx + self.window_size - 1, self.condition_cols].values.astype(np.float32)
        # Target: only the price predictions
        target = self.df.loc[idx + self.window_size, self.target_cols].values.astype(np.float32)
        return torch.tensor(condition), torch.tensor(target)


# -------------------------------
# Naive Zero Model (for Testing)
# -------------------------------
class NaiveZeroModel(nn.Module):
    def __init__(self, target_dim=0):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, x, t, condition):
        if self.target_dim > 0:
            return torch.zeros(condition.shape[0], self.target_dim)
        return torch.zeros_like(x)


# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
def load_data(path, window_size):
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')

    target_tickers = ['NIY', 'NKD', 'CL', 'BZ', 'MES', 'ZN', 'MNQ', 'US']
    reg_exp = '^(date|' + '|'.join(target_tickers) + ')'
    df = df.filter(regex=reg_exp)

    feature_cols = df.filter(regex='(_close|_volume)$').columns.tolist()
    print("Features:", feature_cols)

    df = df.sort_values('date').reset_index(drop=True)

    # Normalize features (using z-score normalization; for close features use pct_change)
    col_norm_factors = {}
    for col in feature_cols:
        if col.endswith('_close'):
            df[col + '_norm'] = df[col].pct_change().fillna(0)
        else:
            df[col + '_norm'] = df[col]
        mean_val = df[col + '_norm'].mean()
        std_val = df[col + '_norm'].std()
        df[col + '_norm'] = (df[col + '_norm'] - mean_val) / std_val
        col_norm_factors[col] = (mean_val, std_val)

    norm_features = [col + '_norm' for col in feature_cols]
    condition_cols = norm_features
    target_cols = [col for col in norm_features if '_close_norm' in col]

    dataset = MultiStockDataset(df, condition_cols, target_cols, window_size)
    # Return dataset and also the number of condition and target features
    return dataset, len(condition_cols), len(target_cols)

# -------------------------------
# Evaluation Functions
# -------------------------------
def calculate_test_loss(model, test_loader, device, timesteps, alphas_bar, loss_fn, extract_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for condition, target in tqdm(test_loader, desc="Testing"):
            condition = condition.to(device)  # full condition (price & volume)
            target = target.to(device)         # price-only target
            batch_size = target.shape[0]
            t = torch.randint(0, timesteps, (batch_size,), device=device)
            a_bar = extract_fn(alphas_bar, t, target.shape)
            noise = torch.randn_like(target)
            noisy_target = torch.sqrt(a_bar) * target + torch.sqrt(1 - a_bar) * noise
            noise_pred = model(noisy_target, t, condition)
            loss = loss_fn(noise_pred, noise)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss

def validate_one_timestep(model_unet, test_loader, device, timesteps, alphas_bar, extract_fn):
    model_unet.eval()
    with torch.no_grad():
        condition, target = next(iter(test_loader))
        condition_sample = condition[0:1].to(device)
        target_sample = target[0:1].to(device)
        t = torch.tensor([timesteps // 2], device=device)
        a_bar = extract_fn(alphas_bar, t, target_sample.shape)
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
    plt.xlabel("Price Feature Index")
    plt.ylabel("Normalized Value")
    plt.title("Validation: Noised Input vs. Diffusion Output vs. Ground Truth (1 Timestep)")
    plt.xticks(feature_indices)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------
# Full Reverse Diffusion Inference Function
# -------------------------------
def full_inference(model, condition, betas, alphas, alphas_bar, timesteps, device, num_target_features):
    """
    Runs full reverse diffusion starting from pure noise.
    Inputs:
      - model: the diffusion model.
      - condition: historical window tensor of shape [B, window_size, num_condition_features].
      - betas, alphas, alphas_bar: diffusion process tensors.
      - timesteps: total number of diffusion steps.
      - device: torch device.
      - num_target_features: number of target features (close prices only).
    Returns:
      - x: final denoised sample tensor of shape [B, num_target_features].
    """
    model.eval()
    batch_size = condition.shape[0]
    x = torch.randn(batch_size, num_target_features, device=device)

    with torch.no_grad():
        for t in reversed(range(1, timesteps)):
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

def evaluate_model(model, test_loader, device, full_diffusion=False, betas=None, alphas=None, alphas_bar=None,
                   timesteps=None, output_features=None):
    """
    Evaluates a model on the test set and returns the MSE loss.
    Args:
        model: The model to evaluate (either diffusion model or naive model)
        test_loader: DataLoader containing test data
        device: torch device
        full_diffusion: bool indicating if this is a diffusion model requiring full inference
        betas, alphas, alphas_bar: diffusion process tensors (only needed if full_diffusion=True)
        timesteps: number of diffusion timesteps (only needed if full_diffusion=True)
        output_features: number of output features (only needed if full_diffusion=True)
    Returns:
        float: overall MSE loss
    """
    mse_loss_fn = torch.nn.MSELoss()
    total_loss = 0.0

    try:
        model.eval()
    except AttributeError:
        pass
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), desc="Evaluating Model") if full_diffusion else enumerate(test_loader)
        for idx, (condition, target) in loop:
            condition = condition.to(device)
            target = target.to(device)

            if full_diffusion:
                pred = full_inference(model, condition, betas, alphas, alphas_bar,
                                      timesteps, device, output_features)
            else:
                pred = model(None, None, condition)

            loss = mse_loss_fn(pred, target)
            total_loss += loss.item()

    overall_mse = total_loss / len(test_loader)
    return overall_mse

def calculate_naive_mse(test_loader, device, target_dim):
    """Calculate MSE for a naive model that always predicts zeros for the target."""
    naive_model = NaiveZeroModel(target_dim=target_dim).to(device)
    overall_mse = evaluate_model(naive_model, test_loader, device, full_diffusion=False)
    return overall_mse

# -------------------------------
# DiffusionTrainer Class: Handles Training & Evaluation
# -------------------------------
class DiffusionTrainer:
    data_path = os.path.join('drive/MyDrive' if 'google.colab' in sys.modules else '..',
                             'Local_Data/focused_futures_30min/interpolated_all_long_term_combo.parquet')
    def __init__(self, epochs, window_size, test_size, batch_size, lr, l1_weight, l2_weight, timesteps,
                 beta_start, beta_end, hidden_dim, base_channels, dropout_rate, model_class, test_full_inf=False):
        self.epochs = epochs
        self.window_size = window_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.lr = lr
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.hidden_dim = hidden_dim
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        self.model_class = model_class
        self.test_full_inference = test_full_inf
        self.dataset, self.input_features, self.output_features = load_data(self.data_path, self.window_size)

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_checkpoint(self, model, filename):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__,
            'hparams': {
                'window_size': self.window_size,
                'input_features': self.input_features,
                'output_features': self.output_features,
                'hidden_dim': self.hidden_dim,
                'base_channels': self.base_channels,
                'dropout_rate': self.dropout_rate,
                'time_steps': self.timesteps,
                'beta_start': self.beta_start,
                'beta_end': self.beta_end,
                'batch_size': self.batch_size
            }
        }
        torch.save(checkpoint, filename)

    @staticmethod
    def load_checkpoint(filepath, device):
        checkpoint = torch.load(filepath, map_location=device)
        model_class = checkpoint['model_class']
        hparams = checkpoint['hparams']

        model = model_class(
            window_size=hparams['window_size'],
            input_features=hparams['input_features'],
            output_features=hparams['output_features'],
            hidden_dim=hparams['hidden_dim'],
            base_channels=hparams['base_channels']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, hparams

    def run_training(self):
        # Data Loading and Preprocessing
        train_size_val = int(len(self.dataset) * (1 - self.test_size))
        test_size_val = len(self.dataset) - train_size_val
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size_val, test_size_val])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Diffusion process parameters
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        betas = betas.to(self.device)
        alphas_bar = alphas_bar.to(self.device)

        # Model, Optimizer, and Loss Setup
        model = self.model_class(self.window_size, self.input_features, self.output_features,
                                 hidden_dim=self.hidden_dim, base_channels=self.base_channels)
        model.to(self.device)

        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2_weight)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        mse_criterion = nn.MSELoss()
        l1_criterion = nn.L1Loss()

        date_string = datetime.datetime.now().strftime('%m-%d-%Y')
        file = f'{type(model).__name__}_best_{date_string}.pth'
        print(f"Training model, saving to {file}")

        if self.test_full_inference:
            naive_mse = calculate_naive_mse(test_loader, self.device, self.output_features)
            print(f"Naive Zero Model MSE on test set (close prices): {naive_mse:.6f}")

        # Training Loop
        best_loss = float('inf')
        for epoch_idx in range(self.epochs):
            epoch = epoch_idx + 1
            model.train()
            epoch_loss = 0.0
            for condition, target in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
                condition = condition.to(self.device)
                target = target.to(self.device)
                batch_size_current = target.shape[0]
                t = torch.randint(0, self.timesteps, (batch_size_current,), device=self.device)
                a_bar = extract(alphas_bar, t, target.shape)
                noise = torch.randn_like(target)
                noisy_target = torch.sqrt(a_bar) * target + torch.sqrt(1 - a_bar) * noise

                noise_pred = model(noisy_target, t, condition)

                l1_loss = torch.tensor(0., device=self.device)
                if self.l1_weight != 0:
                    for param in model.parameters():
                        l1_loss += l1_criterion(param, torch.zeros_like(param))

                loss = mse_criterion(noise_pred, noise) + self.l1_weight * l1_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            test_loss = calculate_test_loss(model, test_loader, self.device,
                                            self.timesteps, alphas_bar, mse_criterion, extract)
            print(f"Epoch {epoch}/{self.epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
            if test_loss < best_loss:
                best_loss = test_loss
                self.save_checkpoint(model, file)

            if self.test_full_inference and (epoch % 10 == 0 or epoch == self.epochs):
                overall_mse = evaluate_model(model, test_loader, self.device,
                                             full_diffusion=True,
                                             betas=betas,
                                             alphas=alphas,
                                             alphas_bar=alphas_bar,
                                             timesteps=self.timesteps,
                                             output_features=self.output_features)
                print(f"Overall MSE on test set (close prices): {overall_mse:.6f}")

        print("Training complete!")
        print("Best Test Loss (Price): {:.4f}".format(best_loss))

if __name__ == '__main__':
    # Optionally, you can test the trainer independently here.
    trainer = DiffusionTrainer(
        epochs=100,
        window_size=64,
        test_size=0.2,
        batch_size=512,
        lr=1e-3,
        l1_weight=1e-5,
        l2_weight=1e-5,
        timesteps=100,
        beta_start=1e-4,
        beta_end=0.02,
        hidden_dim=128,
        base_channels=32,
        dropout_rate=0.2,
        model_class=NaiveZeroModel  # This can be set to a dummy or omitted when running trainer directly.
    )
    trainer.run_training()
