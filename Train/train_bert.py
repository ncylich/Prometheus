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


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device='cuda'):
    model = model.to(device)
    num_tickers = 8  # Sequence length (number of ticker tokens)
    mask_token_id = 8  # ID for the [MASK] token
    mask_prob = 0.15  # Probability of masking a token (15% is common in BERT)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, (x, y, time) in enumerate(train_loader):
            x = x.to(device)  # (B, 2, L, F) where L=num_tickers, F=backcast_size
            time = time.to(device)  # (B, 3) - [hour, month, year]

            batch_size = x.size(0)
            seq_len = x.size(2)  # should be 8
            backcast_size = x.size(-1)

            # Continuous features (price velocities) from the first channel
            cont_feats = x[:, 0, :, :]  # (B, L, backcast_size)
            target_feats = model.normalize_velocities(cont_feats.clone())

            # Create token_ids: [0..7] for each sequence
            token_ids = torch.arange(num_tickers, device=device).unsqueeze(0).expand(batch_size, -1).clone()

            # Randomly mask some tokens
            mask = torch.rand(token_ids.shape, device=device) < mask_prob
            token_ids[mask] = mask_token_id

            # time_indices is simply the provided time info
            time_indices = time  # (B, 3)

            # Forward pass
            predictions = model(token_ids, cont_feats, time_indices)  # (B, L, backcast_size)

            # Compute loss only on masked positions
            mask = mask.unsqueeze(-1).expand_as(target_feats)  # (B,L,F)
            masked_predictions = predictions[mask]
            masked_targets = target_feats[mask]
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
                x = x.to(device)
                time = time.to(device)

                batch_size = x.size(0)
                cont_feats = x[:, 0, :, :]
                target_feats = model.normalize_velocities(cont_feats.clone())

                token_ids = torch.arange(num_tickers, device=device).unsqueeze(0).expand(batch_size, -1).clone()

                # Mask during validation as well to maintain consistency
                mask = torch.rand(token_ids.shape, device=device) < mask_prob
                token_ids[mask] = mask_token_id
                time_indices = time

                # mask shape: (B, L)
                mask = mask.unsqueeze(-1).expand_as(cont_feats)
                cont_feats[mask] = 0  # Zero out masked positions

                predictions = model(token_ids, cont_feats, time_indices)

                masked_predictions = predictions[mask]
                masked_targets = target_feats[mask]

                loss = criterion(masked_predictions, masked_targets)
                val_loss += loss.item()

            val_loss /= len(test_loader)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
