import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

def merge_technical_elliot_wave():
    # Load the technical indicators
    df = pd.read_csv('../Local_Data/technical_indicators.csv')

    # Load the Elliott Wave data
    df_elliot = pd.read_csv('../Local_Data/daily_arrow_counts.csv')
    df_elliot['date'] = df_elliot['day']
    df_elliot = df_elliot.drop(columns=['day'])

    # Combine the two DataFrames on the date index
    merged_df = pd.merge(df, df_elliot, on='date', how='left')

    # replace nans with 0s
    merged_df = merged_df.fillna(0)

    # calculate the difference in price to the next day and store it in "future_price_change"
    merged_df['future_price_change'] = merged_df['CL_close'].shift(-1) - merged_df['CL_close']
    merged_df = merged_df.dropna()

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv('../Local_Data/all_indicators.csv', index=False)


class LinearPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=3, dropout_prob=0.2):
        """
        input_dim: number of input features.
        hidden_dim: number of units in the hidden layers.
        num_layers: total number of layers (including the output layer).
                    For example, num_layers=3 means 2 hidden layers and 1 output layer.
        dropout_prob: dropout probability applied after each hidden layer.
        """
        super(LinearPredictor, self).__init__()
        layers = []
        for i in range(num_layers - 1):
            in_features = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            # Add dropout for regularization.
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
        # Output layer (produces a single continuous value).
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train(model, dataloader, test_dataloader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

        # evaluate the model on the test set
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in test_dataloader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
        test_loss /= len(test_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss:.4f}")

def main():
    # 1. Load and preprocess the CSV data.
    data = pd.read_csv("../Local_Data/all_indicators.csv")
    # Drop the date column.
    data = data.drop(columns=["date"])

    # Separate features (all columns except future_price_change) and the target.
    X = data.drop(columns=["future_price_change"]).values
    y = data["future_price_change"].values.reshape(-1, 1)

    # Normalize every column.
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y)

    # Create a train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_norm, test_size=0.4, random_state=42
    )

    # Convert numpy arrays to torch tensors.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create datasets and dataloaders.
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instantiate the model.
    input_dim = X_train_tensor.shape[1]  # number of features
    model = LinearPredictor(input_dim, hidden_dim=32, num_layers=3, dropout_prob=0.2)

    # Define the loss function and optimizer.
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    # Adding weight_decay applies L2 regularization.
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # 5. Evaluate the model on the test set.
    model.eval()
    test_loss = 0
    predictions_list = []
    actual_list = []
    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            predictions_list.append(outputs)
            actual_list.append(batch_y)
    test_loss /= len(test_dataloader)
    print("Test Loss:", test_loss)

    # 4. Train the model on the training set.
    train(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=1000)

    # Concatenate predictions and actual values.
    predictions_tensor = torch.cat(predictions_list, dim=0)
    actual_tensor = torch.cat(actual_list, dim=0)

    # Inverse transform predictions and actual values to original scale.
    # predictions = scaler_y.inverse_transform(predictions_tensor.numpy())
    # actuals = scaler_y.inverse_transform(actual_tensor.numpy())

    # print("Predictions (original scale):", predictions)
    # print("Actuals (original scale):", actuals)

if __name__ == "__main__":
    main()