import numpy as np
from config import num_class
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import Xception
from tensorflow.keras.regularizers import l2




def train_pretrained_xception(train_generator, test_generator, batch_size, epochs):
    """
    Trains a model based on the pretrained Xception architecture.

    Initializes an Xception model with ImageNet weights, tailored for a binary or ternary classification task.
    The base layers are frozen to retain learned features, and only the top layer is trained. The model is compiled
    with the Adam optimizer and appropriate loss function based on the number of classes.

    Parameters
    ----------
    train_generator : Generator
        The generator yielding batches of training data and labels.
    test_generator : Generator
        The generator yielding batches of testing data and labels.
    batch_size : int
        The batch size for training.
    epochs : int
        The number of epochs for training.

    Returns
    -------
    tuple
        A tuple containing the training history and the trained model.
    """
    # Load the Xception model, pretrained on ImageNet, without the top layer
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base layers to prevent them from being updated during the first phase of training
    for layer in xception_base.layers:
        layer.trainable = False  # Freeze the layer

    # Assemble the overall model for binary or ternary classification
    model = Sequential([
        xception_base,  # The pretrained Xception base
        GlobalAveragePooling2D(),  # Pooling layer to reduce dimensionality
        Dense(3 if num_class != 2 else 2, activation='softmax')  # Output layer adapted for the number of classes
    ])

    # Compile the model with the Adam optimizer and appropriate loss function
    model.compile(optimizer=Adam(), 
                  loss=('categorical_crossentropy' if num_class != 2 else 'binary_crossentropy'),
                  metrics=['accuracy'])

    # Calculate steps per epoch and validation steps based on data size
    steps_per_epoch = max(1, len(train_generator) // batch_size)
    validation_steps = max(1, len(test_generator) // batch_size)

    # Train the model using the provided data generators
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_generator,
        validation_steps=validation_steps,
        epochs=epochs
    )

    # Evaluate the model on the test generator and print out errors and correct predictions
    test_images, test_labels = next(test_generator)  # Obtain a batch of test data
    predictions = model.predict(test_images)  # Predict classes for the test data

    # Convert predictions and actual labels to class indices
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(test_labels, axis=1)

    # Compare the predicted and actual classes, printing errors and correct predictions
    for i, (pred, actual) in enumerate(zip(predicted_classes, actual_classes)):
        if pred != actual:  # Incorrect prediction
            print(f"Error at image {i}: Predicted {pred}, Actual {actual}")
        else:  # Correct prediction
            print(f"Correct at image {i}: Predicted {pred}, Actual {actual}")

    return history, model  # Return the training history and the trained model

def train_pretrained_xception_1(train_generator, test_generator, batch_size, epochs):
    """
    Trains a model based on the pretrained Xception architecture.

    This function initializes the Xception model with ImageNet weights, excludes the top layer,
    and adds new layers for the classification task. It compiles and trains the model using the
    provided data generators. Post-training, it evaluates the model on a batch from the test generator.

    Parameters
    ----------
    train_generator : Generator
        The generator yielding batches of training data and labels.
    test_generator : Generator
        The generator yielding batches of testing data and labels.
    batch_size : int
        The size of the batches for training and validation.
    epochs : int
        The number of epochs for training the model.

    Returns
    -------
    tuple
        A tuple containing the training history and the trained model.
    """
    # Load the pretrained Xception model without the top layer
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers of the Xception base model
    xception_base.trainable = False

    # Assemble the complete model
    model = Sequential([
        xception_base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),  # Additional dense layer with ReLU activation
        Dropout(0.5),  # Dropout layer to reduce overfitting
        Dense(3 if num_class != 2 else 2, activation='softmax')  # Output layer adjusted for the number of classes
    ])

    # Configure the loss function and metrics based on the number of classes
    loss = 'categorical_crossentropy' if num_class != 2 else 'binary_crossentropy'
    model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])

    # Calculate steps per epoch and validation steps
    steps_per_epoch = max(1, len(train_generator) // batch_size)
    validation_steps = max(1, len(test_generator) // batch_size)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_generator,
        validation_steps=validation_steps,
        epochs=epochs
    )

    # Make predictions with the test generator
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images)

    # Convert predictions to class indices
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(test_labels, axis=1)

    # Print comparison of predictions and actual labels
    for i, (pred, actual) in enumerate(zip(predicted_classes, actual_classes)):
        if pred != actual:
            print(f"Incorrect at image {i}: Predicted {pred}, Actual {actual}")
        else:
            print(f"Correct at image {i}: Predicted {pred}, Actual {actual}")

    return history, model


def scheduler(epoch, lr):
    """
    Learning rate scheduler.

    Decreases the learning rate after a certain number of epochs to help the optimizer
    converge. The learning rate is decreased by multiplying with a factor of e^(-0.1)
    after 10 epochs.

    Parameters
    ----------
    epoch : int
        The current training epoch.
    lr : float
        The current learning rate.

    Returns
    -------
    float
        The adjusted learning rate.
    """
    # Decrease learning rate after 10 epochs
    if epoch < 10:
        return lr  # No change if fewer than 10 epochs
    else:
        return lr * tf.math.exp(-0.1)  # Decrease learning rate for later epochs


def train_pretrained_xception_2(train_generator, test_generator, batch_size, epochs):
    """
    Trains a modified Xception model on the provided training and testing data.

    This function initializes the Xception model with ImageNet weights, modifying the top layers
    for the specific task (binary or multi-class classification). The model's top layers are set
    to be trainable for fine-tuning. It is compiled with either RMSprop or Adam optimizer and
    categorical crossentropy loss. The model is trained with data provided by generators.

    Parameters
    ----------
    train_generator : Generator
        The generator that yields batches of training data and labels.
    test_generator : Generator
        The generator that yields batches of testing data and labels.
    batch_size : int
        The size of the batches to use during training.
    epochs : int
        The number of epochs to train the model.

    Returns
    -------
    tuple
        A tuple containing the training history and the trained model.
    """
    # Load the pretrained Xception model, excluding the top layers
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Set the last five layers of the Xception model to be trainable for fine-tuning
    for layer in xception_base.layers[:-5]:
        layer.trainable = False  # Freezing the layer
    for layer in xception_base.layers[-5:]:
        layer.trainable = True  # Unfreezing the layer

    # Construct the full model
    model = Sequential([
        xception_base,  # The pretrained Xception base
        GlobalAveragePooling2D(),  # Pooling layer to reduce dimensionality
        Dense(256, activation='relu'),  # Additional dense layer
        Dropout(0.5),  # Dropout layer to reduce overfitting
        Dense(3 if num_class != 2 else 2, activation='softmax')  # Output layer adjusted for the number of classes
    ])

    # Choose optimizer: RMSprop as an alternative to Adam
    use_rmsprop = True  # Flag to switch between RMSprop and Adam
    optimizer = RMSprop(learning_rate=0.001) if use_rmsprop else Adam(learning_rate=0.001)

    # Compile the model with the specified loss function and optimizer
    loss = 'categorical_crossentropy'  # Use categorical crossentropy for multi-class classification
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Set up the learning rate scheduler
    lr_scheduler = LearningRateScheduler(scheduler)  # Define the learning rate scheduler

    # Train the model using the provided data generators
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_generator) // batch_size),  # Calculate steps per epoch for training
        validation_data=test_generator,
        validation_steps=max(1, len(test_generator) // batch_size),  # Calculate steps for validation
        epochs=epochs,
        callbacks=[lr_scheduler]  # Add the learning rate scheduler as a callback
    )

    # Obtain predictions from the test generator
    test_images, test_labels = next(test_generator)  # Get a batch of test images and labels
    predictions = model.predict(test_images)  # Make predictions with the model

    # Convert predictions and actual labels to class indices
    predicted_classes = np.argmax(predictions, axis=1)  # Convert predictions to class indices
    actual_classes = np.argmax(test_labels, axis=1)  # Convert actual labels to class indices

    # Compare the predictions with the actual labels
    for i, (pred, actual) in enumerate(zip(predicted_classes, actual_classes)):  # Iterate through each prediction
        # Print out the result of each prediction comparison
        if pred != actual:
            print(f"Error at image {i}: Predicted {pred}, Actual {actual}")
        if pred == actual:
            print(f"Correct at image {i}: Predicted {pred}, Actual {actual}")

    return history, model  # Return the training history and the trained model


def train_pretrained_xception_3(train_generator, test_generator, batch_size, epochs):
    """
    Trains a model based on the pretrained Xception architecture.

    This function initializes an Xception model pretrained on the ImageNet dataset, modifies it for 
    the specific binary or multi-class classification task, and trains it using the provided data generators.
    It also includes fine-tuning by making the last few layers trainable and applies L2 regularization and dropout 
    for better generalization.

    Parameters
    ----------
    train_generator : Generator
        The generator yielding batches of augmented training data and labels.
    test_generator : Generator
        The generator yielding batches of augmented testing data and labels.
    batch_size : int
        The batch size for training.
    epochs : int
        The number of epochs for training.

    Returns
    -------
    tuple
        A tuple containing the training history and the trained model.
    """
    # Load the Xception model, pretrained on ImageNet, excluding the top layers
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Load pre-trained Xception model

    # Fine-tuning: make the last few layers trainable
    for layer in xception_base.layers[:(len(xception_base.layers) - 19)]:
        layer.trainable = False  # Freeze the earlier layers
    for layer in xception_base.layers[(len(xception_base.layers) - 19):]:
        layer.trainable = True  # Unfreeze the last 19 layers

    # Build the complete model
    model = Sequential([
        xception_base,  # Pre-trained Xception base
        BatchNormalization(),  # Normalize and scale inputs or activations
        GlobalAveragePooling2D(),  # Pooling layer to reduce spatial dimensions
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),  # Dense layer with L2 regularization
        Dropout(0.7),  # Dropout for regularization
        BatchNormalization(),  # Further normalization
        Dense(3 if num_class != 2 else 2, activation='softmax')  # Output layer for classification
    ])

    # Choose RMSprop as the alternative optimizer to Adam
    use_rmsprop = True  # Flag to switch between RMSprop and Adam optimizers
    optimizer = RMSprop(learning_rate=0.0001) if use_rmsprop else Adam(learning_rate=0.0001)  # Conditional optimizer selection

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model

    # Setup the Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=10, mode='min', verbose=1)  # Learning rate scheduler

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_generator) // batch_size),  # Define steps per epoch
        validation_data=test_generator,
        validation_steps=max(1, len(test_generator) // batch_size),  # Define validation steps
        epochs=epochs,
        callbacks=[lr_scheduler]  # Add learning rate scheduler as callback
    )

    # Get predictions from the test generator
    test_images, test_labels = next(test_generator)  # Retrieve a batch of data from the test generator
    predictions = model.predict(test_images)  # Make predictions on the test images

    # Convert predictions to class indices
    predicted_classes = np.argmax(predictions, axis=1)  # Determine predicted classes
    actual_classes = np.argmax(test_labels, axis=1)  # Determine actual classes

    # Compare predictions with actual labels
    for i, (pred, actual) in enumerate(zip(predicted_classes, actual_classes)):  # Iterate over each prediction and actual label
        if pred != actual:  # Check if the prediction is incorrect
            print(f"Error at image {i}: Predicted {pred}, Actual {actual}")  # Print error case
        if pred == actual:  # Check if the prediction is correct
            print(f"Correct at image {i}: Predicted {pred}, Actual {actual}")  # Print correct case
    return history, model  # Return the training history and the trained model





def train_pretrained_xception_4(train_generator, test_generator, batch_size, epochs):
    """
    Trains a model based on the Xception architecture with fine-tuning and custom top layers.

    This function loads a pre-trained Xception model, unfreezes the top layers for fine-tuning, 
    and adds custom top layers for classification. The model is compiled with RMSprop optimizer 
    and trained using provided data generators. Learning rate reduction and early stopping are 
    applied as callbacks during training.

    Parameters
    ----------
    train_generator : Generator
        The generator yielding batches of augmented training data and labels.
    test_generator : Generator
        The generator yielding batches of augmented testing data and labels.
    batch_size : int
        The batch size for training.
    epochs : int
        The number of epochs for training.

    Returns
    -------
    tuple
        A tuple containing the training history and the trained model.
    """
    # Load the Xception model pre-trained on ImageNet, without the top layer
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Unfreeze some of the top layers for fine-tuning
    for layer in xception_base.layers[:len(xception_base.layers) - 19]:
        layer.trainable = False  # Freeze the layers except for the last 19
    for layer in xception_base.layers[len(xception_base.layers) - 19:]:
        layer.trainable = True   # Unfreeze the last 19 layers

    # Create the full model with custom top layers
    model = Sequential([
        xception_base,  # The pre-trained Xception base
        BatchNormalization(),  # Add BatchNormalization for better convergence
        GlobalAveragePooling2D(),  # Use GlobalAveragePooling to reduce feature dimensions
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),  # A dense layer with L2 regularization
        Dropout(0.7),  # Dropout layer for reducing overfitting
        BatchNormalization(),  # Another BatchNormalization layer
        Dense(num_class, activation='softmax')  # Output layer adjusted for number of classes
    ])

    # Compile the model with RMSprop optimizer and categorical crossentropy loss
    optimizer = RMSprop(learning_rate=0.0001)  # Use RMSprop optimizer with a learning rate of 0.0001
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Set up learning rate scheduler and early stopping
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, mode='max', verbose=1)  # Learning rate reduction on plateau

    # Train the model with the normalized data using the provided data generators
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_generator) // batch_size),  # Calculate steps per epoch based on batch size
        validation_data=test_generator,
        validation_steps=max(1, len(test_generator) // batch_size),  # Calculate validation steps based on batch size
        epochs=epochs,
        callbacks=[lr_scheduler]  # Add callbacks for learning rate adjustment
    )

    # Post-training: evaluate the model with the test data
    test_images, test_labels = next(test_generator)  # Retrieve a batch of test data
    predictions = model.predict(test_images)  # Make predictions on the test images

    # Convert the predictions and actual labels to class indices
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(test_labels, axis=1)

    # Compare predictions with actual labels
    for i, (pred, actual) in enumerate(zip(predicted_classes, actual_classes)):
        if pred != actual:
            print(f"Error at image {i}: Predicted {pred}, Actual {actual}")  # Print errors
        if pred == actual:
            print(f"Correct at image {i}: Predicted {pred}, Actual {actual}")  # Print correct predictions
    return history, model  # Return the training history and the trained model
