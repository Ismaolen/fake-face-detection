import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input


def train_pretrained_vgg16_weighted(normalized_train_data, normalized_test_data, train_generator, test_generator,
                                    train_labels, batch_size=32, epochs=10):
    """
    Trains a VGG16 model with pretrained weights on a dataset, applying class weight adjustments.

    Initializes a VGG16 model with ImageNet weights, modified for binary classification, and
    adjusts for imbalanced classes using computed class weights. The model is compiled with Adam optimizer
    and binary crossentropy loss, then trained with the provided normalized data and generators.

    Parameters
    ----------
    normalized_train_data : tuple
        A tuple containing the training data and labels, both normalized.
    normalized_test_data : tuple
        A tuple containing the testing data and labels, both normalized.
    train_generator : Generator
        The generator yielding batches of augmented training data and labels.
    test_generator : Generator
        The generator yielding batches of augmented testing data and labels.
    train_labels : np.ndarray
        The array of training labels used for computing class weights.
    batch_size : int, optional
        The batch size for training, default is 32.
    epochs : int, optional
        The number of epochs for training, default is 10.

    Returns
    -------
    tuple
        A tuple containing the training history and the trained model.
    """
    weights = compute_class_weights(train_labels)  # Compute class weights for imbalanced data handling
    vgg16 = VGG16(
        include_top=False,  # Exclude the top (fully connected) layers
        weights='imagenet',  # Use weights from training on ImageNet
        input_tensor=Input(shape=(224, 224, 3))  # Define the input tensor shape
    )
    model = Sequential([
        vgg16,  # The pretrained VGG16 base
        Flatten(),  # Flatten the output to feed into the Dense layer
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(
        optimizer='adam',  # Adam optimizer for efficient gradient descent
        loss='binary_crossentropy',  # Loss function suited for binary classification
        metrics=['accuracy']  # Evaluation metric
    )
    history_vgg16_weighted_loss = model.fit(
        train_generator,
        steps_per_epoch=len(normalized_train_data[0]) // batch_size,  # Determine steps per epoch based on data size
        validation_data=test_generator,
        validation_steps=len(normalized_test_data[0]) // batch_size,  # Determine validation steps based on data size
        epochs=epochs,
        class_weight=weights  # Apply the computed class weights
    )
    return history_vgg16_weighted_loss, model  # Return the training history and model


def compute_class_weights(labels):
    """
    Computes class weights for handling imbalanced training data.

    Uses the `balanced` mode to adjust weights inversely proportional to class frequencies
    in the input data. This can be useful for training on imbalanced data, giving more
    importance to underrepresented classes.

    Parameters
    ----------
    labels : np.ndarray
        The array of training labels.

    Returns
    -------
    dict
        A dictionary mapping class indices to their respective weights.
    """
    # Compute the class weights with 'balanced' mode
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return {i: weight for i, weight in enumerate(class_weights)}  # Return the weights as a dictionary mapping class index to weight
