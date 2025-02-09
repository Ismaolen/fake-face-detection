# Fake Face Detection

## Projekt Description
This project implements a Convolutional Neural Network (CNN) to distinguish between fake and real faces, using TensorFlow and Keras for model training. The dataset contains 120,000 images (60,000 real, 60,000 fake), preprocessed to 224x224 pixels. The CNN, based on the Xception model, underwent fine-tuning and data augmentation. Training involved optimizing hyperparameters, while evaluation utilized a Confusion Matrix. A live demo enables real-time predictions with a camera feed. Detailed setup instructions and all code/resources are available in the GitLab repo. This project showcases the effectiveness of deep learning for enhancing facial recognition system security and reliability.

## Clone Project Repository

To start working with the project, the repository needs to be cloned to your local machine using the following command:

```bash
git clone git@github.com:Ismaolen/fake-face-detection.git
```

## Description
This file contains instructions for setting up the virtual environment and running the scripts.

## Setting Up the Virtual Environment

To install dependencies for the project and ensure an isolated development environment, follow these steps:

### Prerequisites

Python 3 should be installed. You can check the Python version with the following command:

```bash
python --version
```

### Creating Virtual Environment

1. Navigate to the project directory:

```bash
cd <path>/fake-face-detection
```

2. Create the virtual environment with the following command:

```bash
python3 -m venv venv
```

### Activating Virtual Environment

- On Windows:

```bash
.\venv\Scripts\activate
```

- On macOS and Linux:

```bash
source venv/bin/activate
```

### Installing Dependencies

Install all required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Note for Windows Users:
TensorFlow version `2.15.0.post1` is required for compatibility with our trained models. However, this version is not directly installable on Windows. Therefore, Windows users have two options:

1. **WSL (Windows Subsystem for Linux)**:
   - Install WSL following the official Microsoft documentation: [Windows Subsystem for Linux Installation Guide for Windows 10](https://docs.microsoft.com/en-us/windows/wsl/install)
   - Once WSL is set up, follow the instructions for non-Windows systems to install dependencies.

2. **Virtual Machine with Linux**:
   - Set up a virtual machine (VM) with Linux installed (e.g., Ubuntu).
   - Install Python and other required dependencies within the Linux VM.
   - Follow the instructions for non-Windows systems to install dependencies.

## Running Scripts
Before running the scripts, the folders apple_red_skalled, real_faces, and Fake_faces from either old_images or neu_images directory need to be extracted into fake_face_detection/data.

After setting up and activating the virtual environment, you can execute the scripts as follows:

```bash
python main.py
```

### NOTE:
For training with the 120,000 images, more computing power and graphics card are required. However, for training with the old data, which consists of the 1000 fake and real images, not as much computing power is needed, but the number of epochs must be adjusted in the config.py file.


## Detection of Real and Fake Images with the Camera using a Pretrained Model

This section uses a video to show how images are recognized as real or fake.

The video cannot be played in Firefox. Please use Google Chrome.

Description of what is seen in the video:

In this video, you will see how images are recognized as real or fake. First, the script is executed, and then various images are presented in front of the camera. The model recognizes the images and determines whether they are real or fake. By the way, all images are labeled as real or fake, so we can determine whether the model has recognized the images correctly or incorrectly.

**Visualization:**

![](./demo/demo_kamera/Video_Demo/Kamera_Demo.mp4){width=1080 height=620}


## Deactivating Virtual Environment

Finally, deactivate the virtual environment with the following command:

```bash
deactivate
```
