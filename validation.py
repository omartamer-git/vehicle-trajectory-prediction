import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

VALIDATION_GENERATED_PLOTS = config['VALIDATION']['VALIDATION_GENERATED_PLOTS']
VALIDATIONSCENE = config['VALIDATION']['VALIDATIONSCENE']
FIGSAVEPATH = config['VALIDATION']['FIGSAVEPATH']
IMUROOT = config['PLOTGEN']['IMUROOT']
MODEL_PATH = config['DEFAULT']['MODEL_SAVE_PATH']
INPUT_LENGTH = 5
PREDICTION_LENGTH = 1

# Constants
ER = 6378137.0  # Average earth radius at the equator

def project_to_image(points_3d, intrinsic_matrix):
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_2d_homogeneous = intrinsic_matrix @ points_3d_homogeneous.T
    points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]
    return points_2d.T

def latlon_to_mercator(lat, lon, scale):
    mx = scale * lon * np.pi * ER / 180
    my = scale * ER * np.log(np.tan((90 + lat) * np.pi / 360))
    return mx, my

def lat_to_scale(lat):
    return np.cos(lat * np.pi / 180.0)

def postprocess_poses(poses_in):
    R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    poses = [np.matmul(R, P.T).T if len(P) else [] for P in poses_in]
    return poses

def convert_oxts_to_pose(oxts, reflat, reflng):
    scale = lat_to_scale(reflat)
    ox, oy = latlon_to_mercator(reflat, reflng, scale)
    origin = np.array([ox, oy, 0])
    tx, ty = latlon_to_mercator(float(oxts[0]), float(oxts[1]), scale)
    t = np.array([tx, ty, float(oxts[2])]) - origin
    rz = float(oxts[5])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return np.vstack((np.hstack((Rz, t.reshape(3, 1))), np.array([0, 0, 0, 1])))

def inverse_pose_transformation(point_global, imu, reflat, reflng):
    pose = convert_oxts_to_pose(imu, reflat, reflng)
    pose = postprocess_poses([pose])[0]
    pose_inv = np.linalg.inv(pose)
    local_point_homogeneous = np.dot(pose_inv, point_global)
    return local_point_homogeneous[:3]

def mean_euclidean_distance(ground_truth, predictions):
    num_steps = len(ground_truth[0])
    total_distances = [0] * num_steps
    count_per_step = [0] * num_steps

    for sequence_index in range(len(ground_truth)):
        for point_index in range(len(ground_truth[sequence_index])):
            gt_point = ground_truth[sequence_index][point_index]
            pred_point = predictions[sequence_index][point_index]
            distance = np.sqrt((gt_point[0] - pred_point[0])**2 + (gt_point[1] - pred_point[1])**2)
            total_distances[point_index] += distance
            count_per_step[point_index] += 1

    mean_distances = [total_distances[i] / count_per_step[i] for i in range(num_steps)]
    return mean_distances

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
                            x, y, _, _ = line.strip().split(',')
                            point = [float(x), float(y)]
                            data.append(point)
                    all_data.append(np.array(data))
    return all_data

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

# Load model
model = TrajectoryPredictor()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Read and process data
raw_data = read_txt_files(VALIDATION_GENERATED_PLOTS)
smoothed_data = []
window_size = 3
poly_order = 2

for trajectory in raw_data:
    if len(trajectory) >= window_size:
        smoothed_x = savgol_filter(trajectory[:, 0], window_size, poly_order)
        smoothed_y = savgol_filter(trajectory[:, 1], window_size, poly_order)
        smoothed_trajectory = np.column_stack((smoothed_x, smoothed_y))
        smoothed_data.append(smoothed_trajectory)

raw_data = smoothed_data
all_points = np.vstack(raw_data)
mean_x = np.mean(all_points[:, 0])
mean_y = np.mean(all_points[:, 1])
std_x = np.std(all_points[:, 0])
std_y = np.std(all_points[:, 1])
z_score_normalized_data = [((trajectory - np.array([mean_x, mean_y])) / np.array([std_x, std_y])) for trajectory in raw_data]

# Prepare data for LSTM
data_X = []
data_Y = []

for trajectory in z_score_normalized_data:
    if len(trajectory) >= INPUT_LENGTH + PREDICTION_LENGTH:
        for i in range(len(trajectory) - INPUT_LENGTH - PREDICTION_LENGTH + 1):
            data_X.append(trajectory[i:i + INPUT_LENGTH])
            data_Y.append(trajectory[i + INPUT_LENGTH:i + INPUT_LENGTH + PREDICTION_LENGTH])

data_X = torch.tensor(data_X, dtype=torch.float32)
data_Y = torch.tensor(data_Y, dtype=torch.float32)
data_X_xy = data_X[:, :, :2]
data_Y_xy = data_Y[:, :, :2]

# Prediction and error calculation
mse_loss = nn.MSELoss()
errors = []

with torch.no_grad():
    predictions = model(data_X_xy)
    for j, (pred, true) in enumerate(zip(predictions, data_Y_xy)):
        error = mse_loss(pred, true)
        errors.append(error.item())
        print(f"Loss {j}: {error.item()}")

average_error = np.mean(errors)
print(f"Average Error: {average_error}")

def denormalize(trajectory, mean_x, mean_y, std_x, std_y):
    mean = np.array([mean_x, mean_y])
    std = np.array([std_x, std_y])
    return trajectory * std + mean

# De-normalize predictions and ground truth data
predictions_denorm = [denormalize(pred.numpy(), mean_x, mean_y, std_x, std_y) for pred in predictions]
data_Y_denorm = [denormalize(gt.numpy(), mean_x, mean_y, std_x, std_y) for gt in data_Y_xy]
data_X_denorm = [denormalize(seq.numpy(), mean_x, mean_y, std_x, std_y) for seq in data_X_xy]

# Plot predictions, ground truth data, and input data
num_plots = len(predictions_denorm)

# Calculate mean Euclidean distance
mean_distances = mean_euclidean_distance(data_Y_denorm, predictions_denorm)
print(f"Mean Euclidean Distance per step: {mean_distances}")

imu_file = np.loadtxt(os.path.join(IMUROOT, f'{VALIDATIONSCENE}.txt'))
pts = []
for i in range(num_plots):
    plt.figure()
    plt.plot(data_X_denorm[i][:, 0], data_X_denorm[i][:, 1], color='purple', label='Input Sequence', linestyle='-', marker='.')
    plt.plot(data_Y_denorm[i][:, 0], data_Y_denorm[i][:, 1], label='Ground Truth', marker='o')
    plt.plot(predictions_denorm[i][:, 0], predictions_denorm[i][:, 1], label='Prediction', marker='x')
    plt.legend()
    plt.title(f'Trajectory Comparison {i + 1}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.savefig(f'{FIGSAVEPATH}/comparison_plot_{i + 1}.png')
    plt.close()
