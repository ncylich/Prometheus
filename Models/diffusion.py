# By end of tonight we will solve quant 🙏.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# Global Hyperparameters
# -------------------------------
epochs = 10
window_size = 60
test_size = 0.2
batch_size = 1024
lr = 1e-3

timesteps = 100  # total diffusion steps
beta_start = 1e-4
beta_end = 0.02


# -------------------------------
# STEP 4: Diffusion Process Setup and Utility Functions
# -------------------------------
def linear_beta_schedule(timesteps, beta_initial=1e-4, beta_final=0.02):
    """
    Creates a linear schedule for beta values (common for DDPM).
    """
    return torch.linspace(beta_initial, beta_final, timesteps)


def extract(a, t, x_shape):
    """
    Extracts coefficients for the given timesteps.

    a: Tensor of shape [timesteps]
    t: Tensor of diffusion timesteps with shape [batch]
    x_shape: Shape of the target tensor
    """
    batch_size = t.shape[0]
    out = a.gather(0, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out


# -------------------------------
# STEP 3: Dataset for Multivariate Time Series
# -------------------------------
class MultiStockDataset(Dataset):
    def __init__(self, df, feature_cols, window_size):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        # Condition: historical window (60 timesteps) for all features
        condition = self.df.loc[idx: idx + self.window_size - 1, self.feature_cols].values.astype(np.float32)
        # Target: next timestep's vector (all features)
        target = self.df.loc[idx + self.window_size, self.feature_cols].values.astype(np.float32)
        return torch.tensor(condition), torch.tensor(target)


# -------------------------------
# STEP 4: Define the Diffusion Model for Multivariate Time Series
# -------------------------------
class DiffusionTimeSeriesModelMulti(nn.Module):
    def __init__(self, window_size, num_features, hidden_dim=64):
        super().__init__()
        # Encode the historical window (condition)
        self.condition_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * num_features, hidden_dim),
            nn.ReLU()
        )
        # Embed the timestep (scalar t)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Main network: combine the noisy target (num_features) with the condition embedding
        self.net = nn.Sequential(
            nn.Linear(num_features + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features)  # Predict noise (same shape as target)
        )

    def forward(self, x, t, condition):
        # x: [batch, num_features] - noisy target
        # t: [batch] - diffusion timestep (integers)
        # condition: [batch, window_size, num_features] - historical window
        cond_emb = self.condition_fc(condition)  # [batch, hidden_dim]
        t = t.float().unsqueeze(1)  # [batch, 1]
        t_emb = self.time_embed(t)  # [batch, hidden_dim]
        combined_emb = cond_emb + t_emb  # [batch, hidden_dim]
        inp = torch.cat([x, combined_emb], dim=1)  # [batch, num_features + hidden_dim]
        noise_pred = self.net(inp)  # [batch, num_features]
        return noise_pred


# -------------------------------
# STEP 5: Naive Baseline Model
# -------------------------------
class NaiveZeroModel(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

    def forward(self, x, t, condition):
        # Simply return a tensor of zeros with the same shape as x.
        return torch.zeros_like(x)


# -------------------------------
# STEP 4.5: Test Loss Calculation Function
# -------------------------------
def calculate_test_loss(model, loader):
    """
    Calculates the test loss along with separate losses for close and volume features.
    """
    model.eval()
    total_loss = 0
    close_loss = 0
    volume_loss = 0
    total_samples = 0

    # Find indices for close and volume features
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
            close_batch_loss = loss_fn(noise_pred[:, close_indices], noise[:, close_indices])
            volume_batch_loss = loss_fn(noise_pred[:, volume_indices], noise[:, volume_indices])

            total_loss += loss.item() * batch_size
            close_loss += close_batch_loss.item() * batch_size
            volume_loss += volume_batch_loss.item() * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_close_loss = close_loss / total_samples
    avg_volume_loss = volume_loss / total_samples
    return avg_loss, avg_close_loss, avg_volume_loss


def calculate_naive_test_loss():
    return calculate_test_loss(naive_model, test_loader)


def validate_one_timestep(model, test_loader, alphas_bar):
    model.eval()
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
        noise_pred = model(noisy_target, t, condition_sample)

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

# -------------------------------
# Main Function
# -------------------------------
def main():
    # -------------------------------
    # Data Loading and Preprocessing
    # -------------------------------
    # Loading df of futures data: ['date', 'CL_open', 'CL_high', 'CL_low', 'CL_close', 'CL_volume', ...]
    df = pd.read_parquet(f'../Local_Data/focused_futures_30min/interpolated_all_long_term_combo.parquet')
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['date'] = df['date'].dt.tz_convert('America/New_York')

    # -------------------------------
    # STEP 1: Select Close and Volume Features (for multiple instruments)
    # -------------------------------
    # Choosing tickers
    target_tickers = ['NIY', 'NKD', 'CL', 'BZ', 'MES', 'ZN', 'MNQ', 'US']
    reg_exp = '^(date|' + '|'.join(target_tickers) + ')'
    df = df.filter(regex=reg_exp)

    # Extract all columns that end with '_close' or '_volume'
    feature_cols = df.filter(regex='(_close|_volume)$').columns.tolist()
    print("Features:", feature_cols)

    # Sort by date and reset index
    df = df.sort_values('date').reset_index(drop=True)

    # -------------------------------
    # STEP 2: Normalize Each Feature Using Z-Score Normalization
    # -------------------------------
    global norm_features  # Define as global so that calculate_test_loss can access it
    col_norm_factors = {}  # Store the mean and std of each column
    for col in feature_cols:
        df[col + '_norm'] = df[col].pct_change().fillna(0) if col.endswith('_close') else df[col]
        mean_val = df[col + '_norm'].mean()
        std_val = df[col + '_norm'].std()
        df[col + '_norm'] = (df[col + '_norm'] - mean_val) / std_val
        col_norm_factors[col] = (mean_val, std_val)

    # List of normalized feature columns
    norm_features = [col + '_norm' for col in feature_cols]

    # -------------------------------
    # STEP 3: Create a Dataset for Multivariate Time Series
    # Each sample: historical window of shape [window_size, num_features]
    # and target: next timestep's vector [num_features]
    # -------------------------------
    dataset = MultiStockDataset(df, norm_features, window_size)
    # Create a DataLoader for the entire dataset (not used in training, kept for reference)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # Create a train/test split
    train_size_val = int(len(dataset) * (1 - test_size))
    test_size_val = len(dataset) - train_size_val
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size_val, test_size_val])
    global train_loader, test_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------
    # STEP 4: Set Up the Diffusion Process and Model
    # -------------------------------
    betas = linear_beta_schedule(timesteps, beta_start, beta_end)  # [timesteps]
    alphas = 1.0 - betas
    global alphas_bar
    alphas_bar = torch.cumprod(alphas, dim=0)

    num_features = len(norm_features)
    model = DiffusionTimeSeriesModelMulti(window_size, num_features)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    global loss_fn
    loss_fn = nn.MSELoss()

    # -------------------------------
    # STEP 4.5: Create Train/Test Split and Test Loss Function
    # -------------------------------
    # (Already created above: train_loader and test_loader)

    # -------------------------------
    # STEP 5: Naive Baseline Model
    # -------------------------------
    global device, naive_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    betas = betas.to(device)
    alphas_bar = alphas_bar.to(device)
    naive_model = NaiveZeroModel(num_features).to(device)

    total_loss, close_loss, volume_loss = calculate_naive_test_loss()
    print(
        f"Naive baseline - Total loss: {total_loss:.4f}, Close loss: {close_loss:.4f}, Volume loss: {volume_loss:.4f}")

    # -------------------------------
    # STEP 6: Training Loop
    #
    # For each sample, we randomly choose a diffusion timestep t,
    # add noise to the target via the closed-form expression,
    # and train the model to predict the added noise.
    # -------------------------------
    total_loss, close_loss, volume_loss = calculate_test_loss(model, test_loader)
    print(
        f"Initial Diffusion Model - Total loss: {total_loss:.4f}, Close loss: {close_loss:.4f}, Volume loss: {volume_loss:.4f}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for condition, target in tqdm(train_loader):
            # Move data to device
            condition = condition.to(device)
            target = target.to(device)
            batch_size_current = target.shape[0]

            # Sample random diffusion timesteps for each sample
            t = torch.randint(0, timesteps, (batch_size_current,), device=device)
            a_bar = extract(alphas_bar, t, target.shape)

            # Sample Gaussian noise and create noisy target
            noise = torch.randn_like(target)
            noisy_target = torch.sqrt(a_bar) * target + torch.sqrt(1 - a_bar) * noise

            # Predict the noise using the model
            noise_pred = model(noisy_target, t, condition)
            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping (optional)
            optimizer.step()

            epoch_loss += loss.item() * batch_size_current

        epoch_loss /= len(train_dataset)
        total_loss, close_loss, volume_loss = calculate_test_loss(model, test_loader)
        validate_one_timestep(model, test_loader, alphas_bar)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, "
              f"Test Loss: Total={total_loss:.4f}, Close={close_loss:.4f}, Volume={volume_loss:.4f}")

    print("Training complete!")


# Only run the main function if this script is executed directly
if __name__ == '__main__':
    main()