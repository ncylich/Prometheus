import sys
import time
import torch
from torch import nn


def process_batch(model, x, device):
    # x = x[:, 0, :, :].to(device)  # (B, sequence_len, num_tickers)
    x = x.permute(0, 1, 3, 2)  # (B, num_tickers, sequence_len)
    recon_x, mu, logvar = model(x)
    return recon_x, x, mu, logvar

def price_mse_loss(recon_x, x):
    x = x[:, 0]
    recon_x = recon_x[:, 0]
    return nn.MSELoss()(recon_x, x)


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs, device='cuda'):
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mse_loss = 0.0
        train_kld_loss = 0.0
        train_price_mse_loss = 0.0

        for i, (x, y, time) in enumerate(train_loader):
            recon_x, x, mu, logvar = process_batch(model, x, device)
            loss, mse, kld = criterion(recon_x, x, mu, logvar)
            price_mse = price_mse_loss(recon_x, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mse_loss += mse.item()
            train_kld_loss += kld.item()
            train_price_mse_loss += price_mse.item()

            # if i % 10 == 0:
            #     print(f"Epoch [{epoch + 1}/{epochs}] - Batch [{i}/{len(train_loader)}] - Train Loss: {loss / (i + 1):.4f} - MSE Loss: {mse / (i + 1):.4f} - KLD Loss: {kld / (i + 1):.4f}")

        train_loss /= len(train_loader)
        train_mse_loss /= len(train_loader)
        train_kld_loss /= len(train_loader)
        train_price_mse_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mse_loss = 0.0
        val_kld_loss = 0.0
        val_price_mse_loss = 0.0

        with torch.no_grad():
            for x, y, time in test_loader:
                recon_x, x, mu, logvar = process_batch(model, x, device)
                loss, mse, kld = criterion(recon_x, x, mu, logvar)
                price_mse = price_mse_loss(recon_x, x)

                val_loss += loss.item()
                val_mse_loss += mse.item()
                val_kld_loss += kld.item()
                val_price_mse_loss += price_mse.item()

            val_loss /= len(test_loader)
            val_mse_loss /= len(test_loader)
            val_kld_loss /= len(test_loader)
            val_price_mse_loss /= len(test_loader)

        if scheduler is not None:
            scheduler.step(val_loss)

        # print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - MSE Loss: {train_mse_loss:.4f} - KLD Loss: {train_kld_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f} - MSE Loss: {train_mse_loss:.4f} - Price MSE Loss: {train_price_mse_loss:.4f} - KLD Loss: {train_kld_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f} - MSE Loss: {val_mse_loss:.4f} - Price MSE Loss: {val_price_mse_loss:.4f} - KLD Loss: {val_kld_loss:.4f}\n")
