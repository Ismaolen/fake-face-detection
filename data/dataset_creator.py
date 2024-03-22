import numpy as np
import matplotlib.pyplot as plt
import h5py
from config import num_class

from utils.logger import print_status


def create_data(real_images, fake_images, other_images=None):
    """
    Creates dataset arrays for training and testing from the input images.

    This function utilizes a DatasetCreator instance to organize real, fake, and optionally other images
    into structured data suitable for model training and testing. It separates the images into
    train and test datasets.

    Parameters
    ----------
    real_images : np.ndarray
        Array of real images to be included in the dataset.
    fake_images : np.ndarray
        Array of fake images to be included in the dataset.
    other_images : np.ndarray, optional
        Array of other category images to be included in the dataset, default is None.

    Returns
    -------
    tuple of np.ndarray
        The training and testing datasets prepared from the input images.
    """
    # Initialize DatasetCreator with the provided image arrays
    dataset_creator = DatasetCreator(real_images, fake_images, other_images)
    # Obtain the training and testing datasets
    train_data, test_data = dataset_creator.get_train_data(), dataset_creator.get_test_data()
    print_status()  # Print the status after creating the datasets
    return train_data, test_data  # Return the structured training and testing datasets


def save_data(real_images, fake_images, other_images=None):
    """
    Saves the input image data arrays into an HDF5 file.

    This function stores the arrays of real, fake, and optionally other images into a new HDF5 file named 'images.h5'.
    The data is saved under separate datasets within the file.

    Parameters
    ----------
    real_images : np.ndarray
        Array of real images to be saved.
    fake_images : np.ndarray
        Array of fake images to be saved.
    other_images : np.ndarray, optional
        Array of other category images to be saved, default is None.

    Returns
    -------
    None
    """
    # Open or create an HDF5 file for writing the image data
    with h5py.File('images.h5', 'w') as hf:  
        hf.create_dataset('real_images', data=real_images)  # Save real images array to the file
        hf.create_dataset('fake_images', data=fake_images)  # Save fake images array to the file
        if num_class != 2:  # Check if there are more than two classes
            hf.create_dataset('other_images', data=other_images)  # Save other images array to the file if applicable
    print_status()  # Print the status after saving the data



class DatasetCreator():
    """
    A class for creating and managing a dataset for machine learning models.

    This class takes arrays of real, fake, and optionally other types of images, assigns labels to them,
    and combines them into a single dataset. It also splits this dataset into training and testing sets.

    Parameters
    ----------
    real_images : np.ndarray
        Array containing real images.
    fake_images : np.ndarray
        Array containing fake images.
    other_images : np.ndarray, optional
        Array containing images of a third class, default is None.
    other_labels : np.ndarray, optional
        Array containing labels for the third class, default is None.

    Attributes
    ----------
    train_data : tuple
        A tuple containing the training data and their corresponding labels.
    test_data : tuple
        A tuple containing the testing data and their corresponding labels.
    """

    def __init__(self, real_images, fake_images, other_images=None, other_labels=None):
        # Assign labels to real and fake images
        real_labels = np.ones(len(real_images))  # Create an array of ones for real images
        fake_labels = np.zeros(len(fake_images))  # Create an array of zeros for fake images
        if num_class != 2:  # Check if there is a third class
            other_labels = np.full(len(other_images), 2)  # Assign a distinct label for the third class

        # Combine images from all classes into one array for faster processing
        if num_class != 2:  # If there are more than two classes
            total_length = len(real_images) + len(fake_images) + len(other_images)
        else:  # If there are only two classes
            total_length = len(real_images) + len(fake_images)
        all_images = np.empty((total_length,) + real_images[0].shape, dtype=real_images[0].dtype)  # Prepare the combined array
        all_images[:len(real_images)] = real_images  # Add real images
        all_images[len(real_images):len(real_images) + len(fake_images)] = fake_images  # Add fake images
        if num_class != 2:
            all_images[len(real_images) + len(fake_images):] = other_images  # Add third class images

        # Combine labels from all classes into one array
        all_labels = np.concatenate([real_labels, fake_labels] +
                                    ([other_labels] if num_class != 2 else []), axis=0)  # Combine all labels

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = self.create_training_and_testing_sets(
            all_images, all_labels, train_size=0.8  # Set 80% of the data as training data
        )

        self.train_data = (x_train, y_train)  # Store training data and labels
        self.test_data = (x_test, y_test)  # Store testing data and labels
        print_status()  # Print the status after creating and splitting the dataset

    def get_train_data(self):
        """
        Returns the training data and labels.

        Returns
        -------
        tuple
            The training data and their corresponding labels.
        """
        print_status()  # Print status before returning training data
        return self.train_data  # Return the training data and labels

    def get_test_data(self):
        """
        Returns the testing data and labels.

        Returns
        -------
        tuple
            The testing data and their corresponding labels.
        """
        print_status()  # Print status before returning testing data
        return self.test_data  # Return the testing data and labels

    def create_training_and_testing_sets(self, x, y, train_size=0.8):
        """
        Splits the data into training and testing sets.

        Parameters
        ----------
        x : np.ndarray
            The array containing the image data.
        y : np.ndarray
            The array containing the corresponding labels.
        train_size : float
            The proportion of the dataset to include in the train split.

        Returns
        -------
        tuple of np.ndarray
            The split training and testing data and their corresponding labels.
        """
        num_train = int(len(x) * train_size)  # Calculate the number of training samples
        indices = np.arange(len(x))  # Create an array of indices
        np.random.shuffle(indices)  # Shuffle the indices

        train_indices = indices[:num_train]  # Indices for the training set
        test_indices = indices[num_train:]  # Indices for the testing set

        x_train = [x[i] for i in train_indices]  # Select the training data
        y_train = [y[i] for i in train_indices]  # Select the training labels
        x_test = [x[i] for i in test_indices]  # Select the testing data
        y_test = [y[i] for i in test_indices]  # Select the testing labels

        print_status()  # Print status after creating the training and testing sets
        return x_train, x_test, y_train, y_test  # Return the split datasets

    def test_data_generator(self, data, num_samples=5):
        """
        Generates and displays a specified number of sample images and labels from the training data.

        This function is primarily used for visual inspection of the training data. It randomly selects a number
        of samples from the provided data and displays each image alongside its label.

        Parameters
        ----------
        data : tuple
            The training data and labels.
        num_samples : int
            The number of samples to display, defaults to 5.

        Returns
        -------
        None
        """
        x_train, y_train = data  # Unpack the training data and labels

        # Ensure the requested number of samples does not exceed the available data
        num_samples = min(num_samples, len(x_train))

        for i in range(num_samples):  # Loop through the specified number of samples
            plt.imshow(x_train[i])  # Display the image
            plt.title(f'Label: {y_train[i]}')  # Display the corresponding label
            plt.show()  # Show the plot
        print_status()  # Print status after generating the sample data

