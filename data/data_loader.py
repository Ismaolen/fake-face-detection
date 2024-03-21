import inspect
import os
import numpy as np
from PIL import Image
import h5py
from config import num_class

from utils.logger import print_status


def load_data(path):
    """
    Load dataset from a specified path.

    This function loads the dataset using a DataLoader from the given path. 
    The dataset is split into different categories based on the number of classes. 
    If there are two classes, it separates the dataset into 'real' and 'fake' images. 
    For more than two classes, it additionally separates 'other' images.

    Parameters
    ----------
    path : str
        The path to the dataset directory.

    Returns
    -------
    tuple of np.ndarray
        The loaded image data, split into categories. Returns two arrays for two classes (real and fake images), 
        and three arrays for more than two classes (real, fake, and other images).
    """
    # Initialize the DataLoader with the provided dataset path
    global other_images  # Declare other_images as a global variable to ensure it is accessible outside the function
    data_loader = DataLoader(dataset_path=path)  # Create an instance of DataLoader to load the dataset

    # Load the dataset and split into categories based on the number of classes
    if num_class == 2:  # Check if there are only two classes
        # Load the dataset and separate into 'real' and 'fake' images
        real_images, fake_images = data_loader.load_dataset()
        print_status()  # Print the current status of data loading
        return real_images, fake_images  # Return the separated 'real' and 'fake' images
    else:
        # Load the dataset and separate into 'real', 'fake', and 'other' images for more than two classes
        real_images, fake_images, other_images = data_loader.load_dataset()
        print_status()  # Print the current status of data loading
        return real_images, fake_images, other_images  # Return the separated 'real', 'fake', and 'other' images



def load_images():
    """
    Loads images from an HDF5 file for model usage.

    Depending on the number of classes defined globally, this function selectively loads 
    images categorized as 'real' and 'fake', and optionally 'other' from a predetermined 
    HDF5 file ('images.h5'). After loading, it prints the current status of the images 
    loaded and returns the image arrays.

    Returns
    -------
    tuple of np.ndarray
        The loaded image arrays categorized as 'real' and 'fake', and optionally 'other' 
        depending on the number of classes. For two classes, returns 'real' and 'fake'. 
        For more than two classes, additionally returns 'other'.
    """
    # Determine the number of classes and load corresponding image data
    if num_class == 2:  # Check if there are exactly two classes
        with h5py.File('images.h5', 'r') as hf:  # Open the HDF5 file containing the images
            real_images = hf['real_images'][:]  # Load 'real' images from the file
            fake_images = hf['fake_images'][:]  # Load 'fake' images from the file
        print_status()  # Print status of image loading
        return real_images, fake_images  # Return the loaded 'real' and 'fake' images
    else:  # If there are more than two classes
        with h5py.File('images.h5', 'r') as hf:  # Open the HDF5 file containing the images
            real_images = hf['real_images'][:]  # Load 'real' images from the file
            fake_images = hf['fake_images'][:]  # Load 'fake' images from the file
            other_images = hf['other_images'][:]  # Load 'other' images from the file
        print_status()  # Print status of image loading
        return real_images, fake_images, other_images  # Return the loaded images categorized into 'real', 'fake', and 'other'



class DataLoader:
    """
    DataLoader is responsible for loading images from specified dataset directories.

    This class is initialized with a path to a dataset. It supports loading images from separate
    directories within the dataset, categorizing them based on their nature (real or fake faces, and potentially others).

    Parameters
    ----------
    dataset_path : str
        The file path to the root of the dataset containing different categories of images.
    """

    def __init__(self, dataset_path):
        print_status()  # Print the initial status of data loading
        self.dataset_path = dataset_path  # Store the dataset path

    def load_dataset(self):
        """
        Loads the dataset from the specified path and categorizes images based on the directory structure.

        Depending on the global `num_class`, it separates images into 'real' and 'fake', and if applicable, 'other'.
        Updates the global 'other_images_path' if more than two classes are considered.

        Returns
        -------
        tuple of np.ndarray
            The images separated into categories ('real', 'fake', and optionally 'other') as Numpy arrays.
        """
        global other_images, other_images_path  # Declare globals for other images and their path
        real_faces_path = os.path.join(self.dataset_path, 'real_faces')  # Path to real faces images
        fake_faces_path = os.path.join(self.dataset_path, 'fake_faces')  # Path to fake faces images
        
        # Conditionally set the path for 'other' images based on the number of classes
        if num_class != 2:
            other_images_path = os.path.join(self.dataset_path, 'apple_red_skalled')

        # Load images from respective folders
        real_images = self.load_images_from_folder(real_faces_path)
        fake_images = self.load_images_from_folder(fake_faces_path)
        if num_class != 2:
            other_images = self.load_images_from_folder(other_images_path)  # Load 'other' images if more than two classes
        print_status()  # Print status after loading images

        # Return the loaded image arrays, including 'other' images if applicable
        if num_class != 2:
            return real_images, fake_images, other_images
        else:
            return real_images, fake_images

    def load_image(self, file_path):
        """
        Loads a single image from a given file path.

        Parameters
        ----------
        file_path : str
            The file path of the image to be loaded.

        Returns
        -------
        np.ndarray or None
            The image as a Numpy array if successfully loaded, None otherwise.
        """
        try:
            with Image.open(file_path) as img:  # Open the image file
                return np.array(img)  # Return the image as a numpy array
        except IOError:
            print(f"Cannot load image: {file_path}")  # Print error if image cannot be loaded
            return None  # Return None if there is an error

    def load_images_from_folder(self, folder):
        """
        Loads all images from a specified folder.

        Parameters
        ----------
        folder : str
            The folder path from which to load all images.

        Returns
        -------
        np.ndarray
            An array of all successfully loaded images from the folder.
        """
        images = []  # Initialize an empty list for storing images
        for filename in os.listdir(folder):  # Iterate over all files in the folder
            img = self.load_image(os.path.join(folder, filename))  # Load each image
            if img is not None:  # Check if the image was successfully loaded
                images.append(img)  # Add the image to the list if it was successfully loaded
        print_status()  # Print status after loading images from the folder
        return np.array(images)  # Return the list of images as a numpy array

