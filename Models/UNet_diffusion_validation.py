import torch
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Models.UNet_diffusion_large import DiffusionTimeSeriesModelUNetLarge
from Train.train_unet_diff import extract, DiffusionTrainer

date = '02-26-2025'
model_class = DiffusionTimeSeriesModelUNetLarge

MODEL_PATH = f'{model_class.__name__}_best_{date}.pth'  # Updated model path

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
        for t in tqdm(reversed(range(1, timesteps)), desc="Reverse Diffusion"):
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
# Dataset for Multivariate Time Series (Separate Condition and Target)
# -------------------------------
class MultiStockDataset(Dataset):
    def __init__(self, df, condition_cols, target_cols, window_size):
        """
        Args:
          - df: DataFrame containing the data.
          - condition_cols: list of columns for the conditioning input (all normalized features).
          - target_cols: list of columns for the target (only normalized close prices).
          - window_size: number of timesteps in the historical window.
        """
        self.df = df.reset_index(drop=True)
        self.condition_cols = condition_cols
        self.target_cols = target_cols
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        # Condition: historical window for all features (close and volume)
        condition = self.df.loc[idx: idx + self.window_size - 1, self.condition_cols].values.astype(np.float32)
        # Target: next timestep's close prices only
        target = self.df.loc[idx + self.window_size, self.target_cols].values.astype(np.float32)
        return torch.tensor(condition), torch.tensor(target)

# -------------------------------
# Naive Zero Model for MSE Calculation
# -------------------------------
class NaiveZeroModel:
    def __init__(self, device, target_dim):
        self.device = device
        self.target_dim = target_dim

    def __call__(self, x, t, condition):
        # Predict zeros for target shape [B, target_dim]
        return torch.zeros(condition.shape[0], self.target_dim, device=self.device)

def calculate_naive_mse(test_loader, device, target_dim):
    """
    Calculate MSE for a naive model that always predicts zeros for the target (close prices).
    """
    naive_model = NaiveZeroModel(device, target_dim)
    mse_loss_fn = torch.nn.MSELoss()
    total_loss = 0.0

    with torch.no_grad():
        for condition, target in tqdm(test_loader, desc='Naive Zero Model'):
            condition = condition.to(device)
            target = target.to(device)
            pred = naive_model(None, None, condition)
            loss = mse_loss_fn(pred, target)
            total_loss += loss.item()

    overall_mse = total_loss / len(test_loader)
    return overall_mse

# -------------------------------
# Main Function
# -------------------------------
def main(model_path=MODEL_PATH):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the larger UNet model
    model, hparams = DiffusionTrainer.load_checkpoint(model_path, device)
    model.to(device)

    # Set up diffusion process tensors using parameters from UNet_diffusion_large
    betas = torch.linspace(hparams['beta_start'], hparams['beta_end'],
                           hparams['time_steps']).to(device)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0).to(device)


    # -------------------------------
    # Load Test Set
    # -------------------------------
    df = pd.read_parquet(DiffusionTrainer.data_path)
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
    # For '_close' columns, use pct_change; for volume, use raw values.
    for col in feature_cols:
        if col.endswith('_close'):
            df[col + '_norm'] = df[col].pct_change().fillna(0)
        else:
            df[col + '_norm'] = df[col]
        mean_val = df[col + '_norm'].mean()
        std_val = df[col + '_norm'].std()
        df[col + '_norm'] = (df[col + '_norm'] - mean_val) / std_val

    # Define condition columns (all normalized features) and target columns (only close prices)
    condition_cols = [col + '_norm' for col in feature_cols]
    target_cols = [col + '_norm' for col in feature_cols if '_close' in col]

    # Create test dataset and dataloader
    dataset = MultiStockDataset(df, condition_cols, target_cols, hparams['window_size'])
    test_size_val = int(len(dataset) * 0.2)
    _, test_dataset = torch.utils.data.random_split(dataset,
                                                    [len(dataset) - test_size_val, test_size_val])
    test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False)

    # -------------------------------
    # Run Full Inference on the Test Set and Calculate MSE
    # -------------------------------
    # Calculate baseline MSE with naive zero model (for close prices)
    naive_mse = calculate_naive_mse(test_loader, device, hparams['output_features'])
    print(f"Naive Zero Model MSE on test set (close prices): {naive_mse:.6f}")

    # Evaluation Loop
    mse_loss_fn = torch.nn.MSELoss()
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for idx, (condition, target) in tqdm(enumerate(test_loader), desc="Evaluating Diffusion Model"):
            condition = condition.to(device)
            target = target.to(device)
            pred = full_inference(
                model, condition, betas, alphas, alphas_bar,
                hparams['time_steps'], device, hparams['output_features']
            )
            loss = mse_loss_fn(pred, target)
            total_loss += loss.item()
            print(f"Running MSE Loss: {total_loss / (idx + 1):.6f}")

    overall_mse = total_loss / len(test_loader)
    print(f"\nFinal Diffusion Model MSE Loss on test set (close prices): {overall_mse:.6f}")


if __name__ == '__main__':
    main()
