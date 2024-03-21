from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import *
from utils.logger import print_status


def create_train_generator(train_data, batch_size_f=batch_size):
    """
    Creates a training data generator for image augmentation.

    Parameters
    ----------
    train_data : tuple(np.ndarray, np.ndarray)
        A tuple containing training data features and labels.
    batch_size_f : int, optional
        The size of the batches to generate. Default is `batch_size`.

    Returns
    -------
    DirectoryIterator
        An iterator over the augmented image data for training.

    Notes
    -----
    The function initializes an `ImageDataGenerator` with specified augmentation
    parameters like rotation, shifting, shear, zoom, flipping, and filling modes.
    It then creates a flow of data from the input arrays.
    """

    # Initialize the image data generator with specified augmentation settings
    train_datagen = ImageDataGenerator(
        rotation_range=rotation_range,  # Degree range for random rotations
        width_shift_range=width_shift_range,  # Range for random horizontal shifts
        height_shift_range=height_shift_range,  # Range for random vertical shifts
        shear_range=shear_range,  # Shearing angle in counter-clockwise direction
        zoom_range=zoom_range,  # Range for random zoom
        horizontal_flip=horizontal_flip,  # Enables random horizontal flips
        vertical_flip=vertical_flip,  # Enables random vertical flips
        fill_mode=fill_mode,  # Points outside boundaries are filled according to given mode
        # channel_shift_range=20.0,  # Range for random channel shifts (commented out as new/optional)
    )
    print_status()

    # Generate and return the training data batches
    return train_datagen.flow(train_data[0], train_data[1], batch_size=batch_size_f)



def create_test_generator(test_data, batch_size_f=batch_size):
    """
    Creates a testing data generator without image augmentation.

    Parameters
    ----------
    test_data : tuple(np.ndarray, np.ndarray)
        A tuple containing testing data features and labels.
    batch_size_f : int, optional
        The size of the batches to generate. Default is `batch_size`.

    Returns
    -------
    DirectoryIterator
        An iterator over the image data for testing.

    Notes
    -----
    The function initializes an `ImageDataGenerator` with no augmentation parameters.
    It's mainly used for generating testing data batches with the original image properties.
    """

    # Initialize the image data generator without any augmentation settings
    test_datagen = ImageDataGenerator()

    print_status()

    # Generate and return the testing data batches
    return test_datagen.flow(test_data[0], test_data[1], batch_size=batch_size_f)



def display_augmented_images(generator, num_samples=5):
    """
    Displays a specified number of augmented images from a generator.

    This function iterates over the provided generator to retrieve and display
    a number of augmented images along with their labels.

    Parameters
    ----------
    generator : Iterator
        An iterator that yields pairs of images and their corresponding labels.
    num_samples : int, optional
        The number of samples to display. Default is 5.

    Returns
    -------
    None
        This function does not return anything but displays images inline.

    Notes
    -----
    Intended for use within Jupyter Notebooks or Python environments capable of
    displaying images. Ensure the matplotlib library is installed and imported.
    """
    
    for i in range(num_samples):  # Iterate over the specified number of samples
        img, label = next(generator)  # Get the next image and label from the generator
        plt.imshow(img[0])  # Display the first image from the batch
        plt.title(f'Label: {label[0]}')  # Set the title as the first label from the batch
        plt.show()  # Display the plotted image     
    print_status()