# Fake Face Detection

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