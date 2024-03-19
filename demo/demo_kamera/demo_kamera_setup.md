# Real vs Fake Face Detection with YOLO Integration

This guide will walk you through the steps to run a live demo for detecting real and fake faces using our custom-trained model in combination with YOLO for face extraction.

## Step 1: YOLO Model Setup

First, you'll need to set up YOLO, which is used to extract human faces from images before passing them to our model for fake or real face detection.

Clone the Darknet repository and build it:
```bash
git clone https://github.com/pjreddie/darknet
cd darknet
make
```

Download the YOLOv3 weights:
```bash
wget https://pjreddie.com/media/files/yolov3.weights
```
Reference: [YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)

## Step 2: Adjust Path in the Script

Next, update the paths in the `live_demo_camera_with_yolo.py` script as follows:

```python
YOLO_CONFIG = '../yolo/darknet/cfg/yolov3.cfg'
YOLO_WEIGHTS = '../yolo/darknet/yolov3.weights'
YOLO_CLASSES = '../yolo/darknet/data/coco.names'
```

## Step 3: Download Our Pre-Trained Model

Download our best-performing model from the HTW Cloud (please request access if necessary). Our model achieved an accuracy of 96% and is identified by the following file name: `b7e2d344-ce54-4ee3-b6c4-db122ef5d5a8.keras`.

## Running the Demo

Ensure you have installed all the requirements listed in the `requirements.txt` and activated your virtual environment.

Execute the script with the command:
```bash
python3 live_demo_camera_with_yolo.py
```

For the slideshow demo, use:
```bash
python3 diashow_with_filenames.py
```

This script displays images in succession, changing every 7 seconds. Point your camera at the screen displaying the images. Our script will then start identifying real and fake faces in these images.

**Important Notes:**
- Ensure you adjust the script paths to match your directory structure.
- Make sure you have the YOLO model and weights correctly set up as per step 1.
- The demo's effectiveness depends on the quality of the camera feed and the lighting conditions.

If you encounter any issues or have questions regarding access to the model files on HTW Cloud, please reach out for assistance.






















