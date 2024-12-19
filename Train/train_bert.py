from time import sleep
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_batch(model, x, time, mask_prob, device):
    num_tickers = 8  # Sequence length (number of ticker tokens)
    mask_token_id = 8  # ID for the [MASK] token

    x = x.to(device)  # (B, 2, L, F) where L=num_tickers, F=backcast_size
    time = time.to(device)  # (B, 3) - [hour, month, year]

    batch_size = x.size(0)
    cont_feats = x[:, 0, :, :]  # (B, L, backcast_size)
    target_feats = cont_feats.clone()

    # Create token_ids: [0..7] for each sequence
    token_ids = torch.arange(num_tickers, device=device).unsqueeze(0).expand(batch_size, -1).clone()

    # Randomly mask some tokens
    mask = torch.rand(token_ids.shape, device=device) < mask_prob
    token_ids[mask] = mask_token_id

    time_indices = time  # (B, 3), time_indices is simply the provided time info

    mask = mask.unsqueeze(-1).expand_as(cont_feats)  # mask shape: (B, L)
    # 80% of masks should be 0, 10% should be random, and 10% should be the same token
    rand_or_same_mask = torch.rand(mask.shape, device=device) < 0.2
    rand_or_same_mask = rand_or_same_mask & mask
    mask = mask ^ rand_or_same_mask

    rand_mask = torch.rand(mask.shape, device=device) < 0.5
    rand_mask = rand_mask & rand_or_same_mask
    cont_feats[mask] = 0  # Zero out masked positions
    cont_feats[rand_mask] = torch.randn_like(cont_feats[rand_mask])  # Randomly replace masked positions

    # Forward pass
    predictions = model(token_ids, cont_feats, time_indices)  # (B, L, backcast_size)

    # Compute loss only on masked positions
    mask = mask.unsqueeze(-1).expand_as(target_feats)  # (B,L,F)
    masked_predictions = predictions[mask]
    masked_targets = target_feats[mask]

    return masked_predictions, masked_targets, mask

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device='cuda'):
    model = model.to(device)
    mask_prob = 0.15  # Probability of masking a token (15% is common in BERT)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, (x, y, time) in enumerate(train_loader):
            masked_predictions, masked_targets, _ = process_batch(model, x, time, mask_prob, device)
            loss = criterion(masked_predictions, masked_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] - Batch [{i}/{len(train_loader)}] - Train Loss: {train_loss / (i + 1):.4f}")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, time in test_loader:
                masked_predictions, masked_targets, _ = process_batch(model, x, time, mask_prob, device)
                loss = criterion(masked_predictions, masked_targets)
                val_loss += loss.item()

            val_loss /= len(test_loader)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
