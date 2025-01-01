import sys
import time
import torch
from torch import nn


def process_batch(model, x, device):
    x = x[:, 0, :, :].to(device)  # (B, sequence_len, num_tickers)
    x = x.permute(0, 2, 1)  # (B, num_tickers, sequence_len)
    recon_x, mu, logvar = model(x)
    return recon_x, x, mu, logvar


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device='cuda'):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for i, (x, y, time) in enumerate(train_loader):
            recon_x, x, mu, logvar = process_batch(model, x, device)
            loss = criterion(recon_x, x, mu, logvar)

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
                recon_x, x, mu, logvar = process_batch(model, x, device)
                loss = criterion(recon_x, x, mu, logvar)
                val_loss += loss.item()

            val_loss /= len(test_loader)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")