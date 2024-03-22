import cv2
import numpy as np

from utils.logger import print_status
from config import num_class
from tensorflow.keras.utils import to_categorical


def scale_images(data, target_size=(224, 224)):
    """
    Scales images to a specified size.

    This function resizes each image in the input array to the given target size using OpenCV's resize function. 
    The target size is set to 224x224 pixels by default.

    Parameters
    ----------
    data : np.ndarray
        The array of images to be scaled.
    target_size : tuple of int
        The desired size (width, height) to which the images should be resized.

    Returns
    -------
    np.ndarray
        The array of scaled images.
    """
    scaled_data = []  # Initialize an empty list to hold the scaled images
    for image in data:  # Loop over each image in the input data
        img = cv2.resize(image, target_size)  # Resize the image to the target size
        scaled_data.append(img)  # Append the resized image to the list of scaled data
    print_status()  # Print status after scaling the images
    return np.array(scaled_data)  # Convert the list of scaled images back to a numpy array and return


def normalize_images(images):
    """
    Normalizes image pixel values to the range 0 to 1.

    This function divides all pixel values in the input images by 255, thus converting 
    them from a range of 0-255 to a range of 0-1, suitable for neural network inputs.

    Parameters
    ----------
    images : np.ndarray
        The array of images to be normalized.

    Returns
    -------
    np.ndarray
        The array of normalized images.
    """
    print_status()  # Print status before normalization
    return images / 255.0  # Normalize the pixel values of the images and return


def preprocess_data(train_images, test_images, train_labels, test_labels):
    """
    Preprocesses the training and testing image and label data.

    This function scales and normalizes image data, and converts labels to one-hot encoding format. 
    It prepares the data for input into a neural network model.

    Parameters
    ----------
    train_images : np.ndarray
        The array of training images.
    test_images : np.ndarray
        The array of testing images.
    train_labels : np.ndarray
        The array of training labels.
    test_labels : np.ndarray
        The array of testing labels.

    Returns
    -------
    tuple
        The preprocessed training and testing data and labels.
    """
    # Scale images to a uniform size of 224x224 pixels for both training and testing sets
    scaled_train_images = scale_images(train_images, (224, 224))
    scaled_test_images = scale_images(test_images, (224, 224))

    # Normalize the images to have pixel values in the range 0 to 1
    normalized_train_images = normalize_images(scaled_train_images)
    normalized_test_images = normalize_images(scaled_test_images)

    # Convert labels into one-hot encoding format based on the number of classes
    train_labels = to_categorical(train_labels, num_classes=(3 if num_class != 2 else 2))
    test_labels = to_categorical(test_labels, num_classes=(3 if num_class != 2 else 2))

    # Compile the normalized data and one-hot encoded labels into tuples for training and testing
    normalized_train_data = (normalized_train_images, train_labels)
    normalized_test_data = (normalized_test_images, test_labels)
    print_status()  # Print status after preprocessing the data
    return normalized_train_data, normalized_test_data  # Return the preprocessed data and labels
