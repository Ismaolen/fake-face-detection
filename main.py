from augmentation.augmenter import create_train_generator, create_test_generator
from config import *
from data.data_loader import load_images, load_data
from data.dataset_creator import create_data, save_data
from data.preprocessor import preprocess_data
from evaluation.confusion_matrix import evaluate_model
from models.xception_pretrained import train_pretrained_xception, train_pretrained_xception_1, \
    train_pretrained_xception_2, train_pretrained_xception_3, train_pretrained_xception_4
from models.vgg16_custom import train_custom_vgg16
from models.vgg16_pretrained import train_pretrained_vgg16
from models.vgg16_pretrained_weighted_loss import train_pretrained_vgg16_weighted
from training_logger.training_logger import document_model_information
import tensorflow as tf


def main():
    """
    Main function to orchestrate the data processing, model training, and evaluation pipeline.

    This function performs several key tasks: loading and preparing the dataset, training a model on the prepared
    dataset, and evaluating the trained model's performance. It handles both binary and multi-class scenarios
    depending on the global `num_class` variable.
    """
    # The code for GPU configuration is commented out. It's used to limit TensorFlow GPU memory usage.
    # Load data from the specified path
    global real_images, fake_images, other_images  # Declare images as global variables for wider accessibility
    if num_class != 2:  # If handling more than two classes (binary classification)
        real_images, fake_images, other_images = load_data('data/')
        # Save the Fake and Real Images
        save_data(real_images, fake_images, other_images)

        real_images, fake_images, other_images = load_images()  # Load images from the source
        train_data, test_data = create_data(real_images, fake_images, other_images)  # Split data into training and testing sets
    else:  # If handling binary classification
        real_images, fake_images = load_data('data/')
        # Save the Fake and Real Images
        save_data(real_images, fake_images)

        real_images, fake_images = load_images()  # Load images for binary classification
        train_data, test_data = create_data(real_images, fake_images)  # Split binary classification data into training and testing sets

    # Extract images and labels from the prepared dataset
    (train_images, train_labels), (test_images, test_labels) = train_data, test_data

    # Preprocess data by scaling and normalizing
    normalized_train_data, normalized_test_data = preprocess_data(train_images, test_images, train_labels, test_labels)

    # Create ImageDataGenerators for training and testing; training generator includes data augmentation
    train_generator = create_train_generator(normalized_train_data)
    test_generator = create_test_generator(normalized_test_data)

    # Train the model using the prepared data and a pretrained Xception network
    history, model = train_pretrained_xception_3(train_generator, test_generator, batch_size, epochs)
    print(history.history)  # Print the training history

    # Document model information and obtain a unique identifier for the model
    unique_id = document_model_information(model, history)

    test_steps = len(test_generator)  # Calculate the number of steps for the test generator
    # Evaluate the trained model's performance
    evaluate_model(model, test_generator, test_steps, unique_id)


if __name__ == "__main__":
    main()

