from config import num_class
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def train_custom_vgg16(train_generator, test_generator, batch_size=32, epochs=10):
    """
    Trains a custom VGG16 model on the provided training and testing data.

    This function constructs a Sequential model resembling VGG16 architecture but adjusts the final Dense layer
    based on the number of classes (binary or ternary). It compiles the model with the Adam optimizer and 
    categorical crossentropy loss. The model is then trained on the provided data generators.

    Parameters
    ----------
    train_generator : Generator
        The generator that yields batches of training data and labels.
    test_generator : Generator
        The generator that yields batches of testing data and labels.
    batch_size : int, optional
        The size of the batches to use during training, default is 32.
    epochs : int, optional
        The number of epochs to train the model, default is 10.

    Returns
    -------
    tuple
        A tuple containing the training history and the trained model.
    """
    # Initialize the model with VGG16 architecture adapted for the specific number of classes
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),  # Convolutional layer with 64 filters
        Conv2D(64, (3, 3), activation='relu', padding='same'),  # Another convolutional layer with 64 filters
        MaxPooling2D((2, 2), strides=(2, 2)),  # MaxPooling to reduce the spatial dimensions
        # Additional typical VGG16 Convolutional and MaxPooling layers would be added here...
        Flatten(),  # Flatten the output to feed into the Dense layers
        Dense(3 if num_class != 2 else 2, activation='softmax')  # Output layer adapted for the number of classes
    ])
    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Calculate the steps per epoch and validation steps
    steps_per_epoch = max(1, len(train_generator) // batch_size)  # Calculate steps per epoch for training
    validation_steps = max(1, len(test_generator) // batch_size)  # Calculate steps for validation

    # Train the model with the training and validation data
    history_vgg16_custom = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_generator,
        validation_steps=validation_steps,
        epochs=epochs
    )
    return history_vgg16_custom, model  # Return the training history and the trained model

