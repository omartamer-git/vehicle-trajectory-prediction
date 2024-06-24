import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import savgol_filter
import configparser

# Configuration
config = configparser.ConfigParser()
config.read('config.ini')
TXT_DIR = config['DEFAULT']['TXT_DIR']
MODEL_SAVE_PATH = config['DEFAULT']['MODEL_SAVE_PATH']

# Constants
INPUT_LENGTH = 5
PREDICTION_LENGTH = 1
WINDOW_SIZE = 3  # Must be odd
POLY_ORDER = 2  # Polynomial order for Savitzky-Golay filter
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_SPLIT_RATIO = 0.8

# Function to read data from text files
def read_txt_files(base_path):
    all_data = []
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.txt'):
                    file_path = os.path.join(folder_path, file)
                    data = []
                    with open(file_path, 'r') as f:
                        for line in f:
                            x, y, _, __ = map(float, line.strip().split(','))
                            data.append([x, y])
                    all_data.append(np.array(data))
    return all_data

# Define the LSTM model for trajectory prediction
class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=196, num_layers=1):
        super(TrajectoryPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, PREDICTION_LENGTH * 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.linear(last_out)
        return output.reshape(-1, PREDICTION_LENGTH, 2)

# Model instantiation
model = TrajectoryPredictor()

# Data Preparation
raw_data = read_txt_files(TXT_DIR)

# Applying Savitzky-Golay filter
smoothed_data = []
for trajectory in raw_data:
    if len(trajectory) >= WINDOW_SIZE:
        smoothed_x = savgol_filter(trajectory[:, 0], WINDOW_SIZE, POLY_ORDER)
        smoothed_y = savgol_filter(trajectory[:, 1], WINDOW_SIZE, POLY_ORDER)
        smoothed_trajectory = np.column_stack((smoothed_x, smoothed_y))
        smoothed_data.append(smoothed_trajectory)

# Normalization
all_points = np.vstack(smoothed_data)
mean_x, mean_y = np.mean(all_points, axis=0)
std_x, std_y = np.std(all_points, axis=0)
z_score_normalized_data = [(trajectory - [mean_x, mean_y]) / [std_x, std_y] for trajectory in smoothed_data]

# Prepare data for LSTM
data_X, data_Y = [], []
for trajectory in z_score_normalized_data:
    if len(trajectory) >= INPUT_LENGTH + PREDICTION_LENGTH:
        for i in range(len(trajectory) - INPUT_LENGTH - PREDICTION_LENGTH + 1):
            data_X.append(trajectory[i:i + INPUT_LENGTH])
            data_Y.append(trajectory[i + INPUT_LENGTH:i + INPUT_LENGTH + PREDICTION_LENGTH])

data_X = torch.tensor(data_X, dtype=torch.float32)
data_Y = torch.tensor(data_Y, dtype=torch.float32)

# Split data into training and validation sets
num_samples = data_X.shape[0]
train_size = int(num_samples * TRAIN_SPLIT_RATIO)
train_X, val_X = data_X[:train_size], data_X[train_size:]
train_Y, val_Y = data_Y[:train_size], data_Y[train_size:]

# Creating datasets and loaders
train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=BATCH_SIZE, shuffle=False)

# Setup optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}')

# Validation function
def validate():
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss / len(val_loader):.4f}')

# Run the training process
for epoch in range(NUM_EPOCHS):
    train(epoch)
    validate()

# Save the model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
