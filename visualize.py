import os, numpy as np, sys, cv2, glob
import random
from time import sleep
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

scene = input('Input scene number (0000-0020): ')
dataroot = config['PLOTGEN']['DATAROOT']
display_txt = config['PLOTGEN']['LABELS']

def read_tracking_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        return lines

trkColors = {}
def draw_on_img(img, splits):
    trk_id = splits[1]
    if(int(splits[3]) == 0 or int(splits[3]) == 1):
        pass


    x_min = int(float(splits[6]))
    y_min = int(float(splits[7]))
    x_max = int(float(splits[8]))
    y_max = int(float(splits[9]))

    if(trk_id not in trkColors):
        color = [random.randint(0, 255) for _ in range(3)]
        trkColors[trk_id] = color

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), trkColors[trk_id], thickness = 1)
    # add track id
    tf = 2
    label = f"ID: {trk_id}"
    t_size = cv2.getTextSize(str(label), 0, fontScale = 1, thickness=tf)
    print(t_size)
    c2 = x_min + t_size[0][0], x_max - t_size[0][1] - 3
    # cv2.rectangle(img, (x_min, y_min), c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, str(label), (x_min, y_min - 2), 0, 1, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


lines = read_tracking_file(display_txt + scene + '.txt')

frame_size = (1242, 375) # Example frame size, adjust to your dataset
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, frame_size)


frame = 0
padded_string = str(frame).zfill(6)
img = cv2.imread(dataroot + scene + '/' + padded_string + '.png')
for line in lines:
    splits = line.split(' ')
    frame_no = int(splits[0])
    if(splits[2] != "Car" and splits[2] != "Van" and splits[2] != "Truck"):
        continue
    if(frame == frame_no):
        draw_on_img(img, splits)
    else:
        frame = frame + 1
        padded_string = str(frame).zfill(6)
        # out.write(img)
        cv2.imshow("TR", img)
        cv2.waitKey(150)
        img = cv2.imread(dataroot + scene + '/' + padded_string + '.png')
out.release()
cv2.destroyAllWindows()
