# Vehicle Trajectory Prediction

## Overview
This project is aimed at predicting vehicle trajectories using the KITTI dataset. The project includes scripts for data preparation, model training, validation, and visualization.

## Requirements
To get started, you'll need to install the required Python packages. You can do this by running:

```sh
pip install -r requirements.txt
```

## Dataset
Download the KITTI dataset from [KITTI's official website](http://www.cvlibs.net/datasets/kitti/) and organize it as follows:

```
/path/to/KITTI
    ├── training
    │   ├── image_02
    │   ├── label_02
    │   ├── oxts
```

Alternatively, for a faster setup, you can split KITTI's training data.

## Data Preparation
Generate 2D/3D detections using YOLO and MMDetection3D, and generate MOTs tracking using DeepFusionMOT. 

## Configuration
Adjust the `config.ini` file to specify the paths for your data and model settings. Below is an example configuration:

```ini
[DEFAULT]
TXT_DIR=/home/username/plots/
MODEL_SAVE_PATH=/home/username/vehicle-trajectory-prediction/model.pt

[PLOTGEN]
DATAROOT=/path/to/KITTI/training/image_02/
LABELS=/path/to/KITTI/training/label_02/
IMUROOT=/path/to/KITTI/training/oxts/
PLOTROOT=/home/username/plots/

[VALIDATION]
VALIDATIONSCENE=0002
VALIDATION_GENERATED_PLOTS=/home/username/plots_val
FIGSAVEPATH=/home/username/plots_vali
```

### Config Parameters
- **TXT_DIR**: Directory where generated plot TXT files will be saved.
- **MODEL_SAVE_PATH**: Path to save the trained model.
- **DATAROOT**: Root directory of the KITTI images.
- **LABELS**: Directory containing KITTI label files.
- **IMUROOT**: Directory containing KITTI oxts files.
- **PLOTROOT**: Directory to save generated plots.
- **VALIDATIONSCENE**: Specific scene number for validation.
- **VALIDATION_GENERATED_PLOTS**: Directory to save validation plots.
- **FIGSAVEPATH**: Directory to save validation figures.

## Steps to Run the Project

### 1. Generate Plots
Run the `generate_plots.py` script to convert KITTI labels to the format used by our model.

```sh
python generate_plots.py
```

### 2. Train the Model
Run `train.py` to train the model on the prepared data.

```sh
python train.py
```

### 3. Validate the Model
Run `validate.py` to validate the model.

```sh
python validate.py
```

### 4. Visualize Results
Use `visualize.py` to overlay the labels on the input training images and create a video visualization.

```sh
python visualize.py
```

## Pretrained Models
You can find pretrained models in the `Pretrained` directory. The file names follow the format:
```
Input Coordinates Count-Output Coordinates Count-Neurons Per Layer-Number of Layers.pt
```

## Scripts

### generate_plots.py
This script processes the KITTI dataset to generate the required TXT files for training.

### train.py
This script trains the LSTM model for vehicle trajectory prediction.

### validate.py
This script validates the trained model using a validation dataset.

### visualize.py
This script generates a video showing the labels overlaid on the input images for visualization.

## Conclusion
This project provides a comprehensive pipeline for vehicle trajectory prediction using the KITTI dataset. By following the steps outlined above, you can train, validate, and visualize your own trajectory prediction model.
