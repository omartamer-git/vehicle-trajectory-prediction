import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

dataroot = config['PLOTGEN']['DATAROOT']
display_txt = config['PLOTGEN']['LABELS']
imuroot = config['PLOTGEN']['IMUROOT']
plotroot = config['PLOTGEN']['PLOTROOT']
er = 6378137. # average earth radius at the equator

def latlonToMercator(lat, lon, scale):
    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))
    return mx, my

def latToScale(lat):
    return np.cos(lat * np.pi / 180.0)

def postprocessPoses(poses_in):
    R = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    poses = []
    for i in range(len(poses_in)):
        if not len(poses_in[i]):
            poses.append([])
            continue
        P = poses_in[i]
        poses.append(np.matmul(R, P.T).T)
    return poses

def convertOxtsToPose(oxts, reflat, reflng):
    scale = latToScale(reflat)
    ox, oy = latlonToMercator(reflat, reflng, scale)
    origin = np.array([ox, oy, 0])
    tx, ty = latlonToMercator(float(oxts[0]), float(oxts[1]), scale)
    t = np.array([tx, ty, float(oxts[2])]) - origin
    rz = float(oxts[5])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return np.vstack((np.hstack((Rz, t.reshape(3, 1))), np.array([0, 0, 0, 1])))

scenes = [d for d in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, d))]
for scene in scenes:
    scene_path = os.path.join(dataroot, scene)
    label_file = os.path.join(display_txt, scene + '.txt')
    imu_file = np.loadtxt(os.path.join(imuroot, scene + '.txt'))
    scene_plots_path = os.path.join(plotroot, scene)
    os.mkdir(scene_plots_path)


    with open(label_file, 'r') as file:
        lines = file.readlines()

    tracking_ids = set(int(line.split(' ')[1]) for line in lines)
    tracking_ids = set(int(line.split(' ')[1]) for line in lines if (line.split(' ')[2] == 'Car' or line.split(' ')[2] == 'Van' or line.split(' ')[2] == 'Truck'))

    for track_id in tracking_ids:
        points = []
        points_ego = []
        F = True
        reflat, reflng = 102309120,102309120
        for line in lines:
            splits = line.split(' ')
            frame_no = int(splits[0])
            tid = int(splits[1])

            if tid == track_id:
                imu = imu_file[frame_no]
                if F:
                    F = False
                    # reflat, reflng = 0,0
                    reflat, reflng = float(imu[0]), float(imu[1])
                pose = convertOxtsToPose(imu, reflat, reflng)
                pose = postprocessPoses([pose])[0]
                x, y, z = float(splits[15]), float(splits[13]), float(splits[14])
                xe, ye, ze = 0,0,0
                point_global = np.dot(pose, np.array([x, y, z, 1]))
                point_ego_global = np.dot(pose, np.array([xe, ye, ze, 1]))
                print(point_global)
                points.append([point_global[0], point_global[1], point_global[2], point_global[3]])
                points_ego.append([point_ego_global[0], point_ego_global[1], point_ego_global[2], point_ego_global[3]])

        if points:
            x, y, _z, _one = zip(*points)
            colors = [[i / len(x), 0, 0] for i in range(len(x))]

            x_ego, y_ego, _z_ego, _one_ego = zip(*points_ego)
            colors2 = [[0, i / len(x_ego), 0] for i in range(len(x_ego))]

            plt.scatter(x, y, c=colors)
            plt.scatter(x_ego, y_ego, c=colors2)
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title(f'Scatter Graph for {scene} - Track ID {track_id}')
            plot_dir = os.path.expanduser(os.path.join(plotroot, scene))
            with open(os.path.join(plot_dir, f'{track_id}.txt'), 'w') as f:
                # Iterate over each point in the 2D array
                for point in points:
                    # Convert each point to a string in the format "x,y"
                    point_str = ','.join(map(str, point))
                    # Write the formatted string to the file, followed by a newline
                    f.write(point_str + '\n')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'{track_id}.png'))
            plt.close()
