# Fake Face Detection

## Projekt Description
This project implements a Convolutional Neural Network (CNN) to distinguish between fake and real faces, using TensorFlow and Keras for model training. The dataset contains 120,000 images (60,000 real, 60,000 fake), preprocessed to 224x224 pixels. The CNN, based on the Xception model, underwent fine-tuning and data augmentation. Training involved optimizing hyperparameters, while evaluation utilized a Confusion Matrix. A live demo enables real-time predictions with a camera feed. Detailed setup instructions and all code/resources are available in the GitLab repo. This project showcases the effectiveness of deep learning for enhancing facial recognition system security and reliability.

## Description
This file contains instructions for setting up the virtual environment and running the scripts.

## Clone Project Repository

To start working with the project, the repository needs to be cloned to your local machine using the following command:

```bash
git clone https://gitlab.rz.htw-berlin.de/s0580078/fake_face_detection.git
```

## Setting Up the Virtual Environment

To install dependencies for the project and ensure an isolated development environment, follow these steps:

### Prerequisites

Python 3.11 should be installed. You can check the Python version with the following command:

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
python3.11 -m venv venv
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

After setting up and activating the virtual environment, you can execute the scripts as follows:

```bash
python main.py
```

## Deactivating Virtual Environment

Finally, deactivate the virtual environment with the following command:

```bash
deactivate
```