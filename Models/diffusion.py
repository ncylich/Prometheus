# By end of tonight we will solve quant 🙏.
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Loading df
# df of futures data: ['date', 'CL_open', 'CL_high', 'CL_low', 'CL_close', 'CL_volume', ...]
df = pd.read_parquet(f'../Local_Data/focused_futures_30min/interpolated_all_long_term_combo.parquet')
df['date'] = pd.to_datetime(df['date'], utc=True)
df['date'] = df['date'].dt.tz_convert('America/New_York')

# -------------------------------
# STEP 1: Select Close and Volume features (for multiple instruments)
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
# STEP 2: Normalize each feature using z-score normalization
# -------------------------------
for col in feature_cols:
    df[col + '_norm'] = df[col].pct_change().fillna(0) if col.endswith('_close') else df[col]
    mean_val = df[col + '_norm'].mean()
    std_val = df[col + '_norm'].std()
    df[col + '_norm'] = (df[col + '_norm'] - mean_val) / std_val

# List of normalized feature columns
norm_features = [col + '_norm' for col in feature_cols]

# -------------------------------
# STEP 3: Create a dataset for multivariate time series
# Each sample: historical window of shape [window_size, num_features]
# and target: next timestep's vector [num_features]
# -------------------------------
window_size = 60


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


dataset = MultiStockDataset(df, norm_features, window_size)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)


# -------------------------------
# STEP 4: Set up the diffusion process and model
#
# We define a linear beta schedule (common for DDPM) for timesteps,
# and build a simple conditional diffusion model that predicts noise.
# -------------------------------
def linear_beta_schedule(timesteps):
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


timesteps = 100  # total diffusion steps
betas = linear_beta_schedule(timesteps)  # [timesteps]
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)


def extract(a, t, x_shape):
    # a: [timesteps], t: [batch], x_shape: target shape
    batch_size = t.shape[0]
    out = a.gather(0, t).reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out


# Define a basic diffusion model for multivariate time series
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


num_features = len(norm_features)
model = DiffusionTimeSeriesModelMulti(window_size, num_features)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# -------------------------------
# STEP 4.5: Create train/test split and test loss function
# -------------------------------
test_size = 0.2
train_size = int(len(dataset) * (1 - test_size))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)


def calculate_test_loss():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for condition, target in tqdm(test_loader):
            condition = condition.to(device)
            target = target.to(device)
            batch_size = target.shape[0]

            # Sample random diffusion timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device)
            a_bar = extract(alphas_bar, t, target.shape)

            # Add noise
            noise = torch.randn_like(target)
            noisy_target = torch.sqrt(a_bar) * target + torch.sqrt(1 - a_bar) * noise

            # Predict noise
            noise_pred = model(noisy_target, t, condition)
            loss = loss_fn(noise_pred, noise)
            total_loss += loss.item() * batch_size

    return total_loss / len(test_dataset)


# -------------------------------
# STEP 5: Training loop
#
# For each sample, we randomly choose a diffusion timestep t,
# add noise to the target via the closed-form expression,
# and train the model to predict the added noise.
# -------------------------------
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
betas = betas.to(device)
alphas_bar = alphas_bar.to(device)

print(f"Initial test loss: {calculate_test_loss():.4f}")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for condition, target in tqdm(train_loader):
        # Move data to device
        condition = condition.to(device)
        target = target.to(device)
        batch_size = target.shape[0]

        # Sample random diffusion timesteps for each sample
        t = torch.randint(0, timesteps, (batch_size,), device=device)
        a_bar = extract(alphas_bar, t, target.shape)

        # Sample Gaussian noise and create noisy target
        noise = torch.randn_like(target)
        noisy_target = torch.sqrt(a_bar) * target + torch.sqrt(1 - a_bar) * noise

        # Predict the noise using the model
        noise_pred = model(noisy_target, t, condition)
        loss = loss_fn(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        epoch_loss += loss.item() * batch_size

    epoch_loss /= len(train_dataset)
    test_loss = calculate_test_loss()
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")

print("Training complete!")