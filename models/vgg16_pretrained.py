from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input


def train_pretrained_vgg16(normalized_train_data, normalized_test_data, train_generator, test_generator, batch_size=32,
                           epochs=10):
    """
    Trains a model based on the pretrained VGG16 architecture for binary classification.

    This function initializes the VGG16 model with ImageNet weights, excluding the top layer, and adds a new top layer 
    for binary classification. The model is compiled with the Adam optimizer and binary crossentropy loss. It is then 
    trained using the provided data generators.

    Parameters
    ----------
    normalized_train_data : tuple
        A tuple containing the training data and labels, both normalized.
    normalized_test_data : tuple
        A tuple containing the testing data and labels, both normalized.
    train_generator : Generator
        The generator that yields batches of augmented training data and labels.
    test_generator : Generator
        The generator that yields batches of augmented testing data and labels.
    batch_size : int, optional
        The size of the batches to use during training, default is 32.
    epochs : int, optional
        The number of epochs to train the model, default is 10.

    Returns
    -------
    tuple
        A tuple containing the training history and the trained model.
    """
    # Load the pretrained VGG16 model without the top layer and with ImageNet weights
    vgg16 = VGG16(
        include_top=False,  # Exclude the top layer
        weights='imagenet',  # Use ImageNet weights
        input_tensor=Input(shape=(224, 224, 3))  # Specify the input shape
    )
    # Define the new top layer and compile the model
    model = Sequential([
        vgg16,  # The pretrained VGG16 base
        Flatten(),  # Flatten the output to feed into the Dense layer
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    # Compile the model with binary settings for a binary classification task
    model.compile(
        optimizer='adam',  # Adam optimizer
        loss='binary_crossentropy',  # Binary crossentropy loss
        metrics=['accuracy']  # Accuracy metric
    )
    # Train the model with the normalized data using the provided data generators
    history_vgg16_pretrained = model.fit(
        train_generator,
        steps_per_epoch=len(normalized_train_data[0]) // batch_size,  # Calculate steps per epoch for training
        validation_data=test_generator,
        validation_steps=len(normalized_test_data[0]) // batch_size,  # Calculate steps for validation
        epochs=epochs  # Number of training epochs
    )
    return history_vgg16_pretrained, model  # Return the training history and the trained model
