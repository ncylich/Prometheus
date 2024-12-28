import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
import random

def process_batch(model, x, time, mask_prob, device):
    num_tickers = x.size(-2)  # Sequence length (number of ticker tokens), -2nd dim of X
    mask_token_id = num_tickers  # ID for the [MASK] token

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
    zero_mask = mask ^ rand_or_same_mask

    rand_mask = torch.rand(mask.shape, device=device) < 0.5
    rand_mask = rand_mask & rand_or_same_mask
    cont_feats[zero_mask] = 0  # Zero out masked positions
    cont_feats[rand_mask] = torch.randn_like(cont_feats[rand_mask])  # Randomly replace masked positions

    # Forward pass
    predictions = model(token_ids, cont_feats, time_indices)  # (B, L, backcast_size)

    # Compute loss only on masked positions
    masked_predictions = predictions[mask]
    masked_targets = target_feats[mask]

    return masked_predictions, masked_targets

class NaiveModel(nn.Module):
    def __init__(self, zeros=False, same_as_input=False):
        assert zeros ^ same_as_input, "zeros or same_as_input must be true, but not both"
        super(NaiveModel, self).__init__()
        self.zeros = zeros
        self.same_as_input = same_as_input

    def forward(self, token_ids, cont_feats, time_indices):
        if self.zeros:
            return torch.zeros_like(cont_feats)
        if self.same_as_input:
            return cont_feats

class NoOpOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = []

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none=False):
        pass

class CriterionNoBackward(nn.Module):
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion

    def forward(self, predictions, targets):
        loss = self.base_criterion(predictions, targets)
        loss.requires_grad_(True)  # Ensure the loss tensor requires gradients
        return loss

    def backward(self, predictions, targets):
        pass

def train_naive_models(train_loader, test_loader, criterion, device='cuda'):
    start = time.time()

    optimizer = NoOpOptimizer()
    criterion = CriterionNoBackward(criterion)

    print("Training The Naive Zero-Model")
    model = NaiveModel(zeros=True).to(device)
    train_model_base(model, train_loader, test_loader, criterion, optimizer, None, epochs=1, device=device)
    print("X" * 60)

    print("Training The Naive Duplicating-Input-Model")
    model = NaiveModel(same_as_input=True).to(device)
    train_model_base(model, train_loader, test_loader, criterion, optimizer, None, epochs=1, device=device)
    print("X" * 60)

    print(f"Training naive models took {time.time() - start:.2f} seconds\n")


def train_model_base(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device='cuda'):
    model = model.to(device)
    mask_prob = 0.15  # Probability of masking a token (15% is common in BERT)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, (x, y, time) in enumerate(train_loader):
            masked_predictions, masked_targets = process_batch(model, x, time, mask_prob, device)
            loss = criterion(masked_predictions, masked_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if (i + 1) % 10 == 0 or i + 1 == len(train_loader):
                print(f"Epoch [{epoch + 1}/{epochs}] - Batch [{i + 1}/{len(train_loader)}] - Train Loss: {train_loss / (i + 1):.4f}")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, time in test_loader:
                masked_predictions, masked_targets = process_batch(model, x, time, mask_prob, device)
                loss = criterion(masked_predictions, masked_targets)
                val_loss += loss.item()

            val_loss /= len(test_loader)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device='cuda'):
    train_naive_models(train_loader, test_loader, criterion, device)  # Get a baseline before training
    return train_model_base(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device)