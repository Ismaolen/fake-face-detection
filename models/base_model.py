
import tensorflow as tf


def build_model_1(input_dim):
    """
    Constructs a multi-layer perceptron (MLP) model without hidden layers, which effectively
    makes it a logistic regression model due to the single output neuron with a sigmoid activation.

    Parameters:
    ----------
    input_dim: int
        The number of features in the input data. This parameter sets the size of the input layer,
        where each feature is represented by one input neuron.

    Returns:
    ---------
    model : Sequential
        The compiled TensorFlow Keras MLP model. This model is suitable for binary classification tasks.
    """
    # Initialize the model as a sequential model. This type of model is a linear stack of layers.
    model = tf.keras.models.Sequential([
        # Add the output layer with a single neuron since this is a binary classification task.
        # 'input_dim' specifies the size of the input layer.
        # No hidden layers are used in this model, which implies that the model is a simple linear classifier.
        tf.keras.layers.Dense(1, input_dim=input_dim)  # Output layer with 1 neuron
    ])

    # Compile the model with the 'adam' optimizer. Adam is an algorithm for gradient-based optimization of stochastic
    # objective functions, designed to be robust to the choice of hyperparameters. The loss function is
    # 'BinaryCrossentropy', which is suitable for binary classification problems.
    # The model is using 'accuracy' as a metric for performance evaluation during training and validation.
    model.compile(optimizer="adam",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # Return the compiled model. The model is now ready to be trained with a dataset.
    return model


def build_model_2(input_dim):
    """
    Constructs a multi-layer perceptron (MLP) model with one hidden layer for binary classification.

    Parameters:
    ----------
    input_dim: int
        The number of input features. This determines the size of the input layer,
        where each feature corresponds to one input neuron.

    Returns:
    ---------
    model : Sequential
        The TensorFlow Keras MLP model that has been compiled with the specified settings.
    """
    # Initialize the model as a sequential model, which is a linear stack of layers.
    model = tf.keras.models.Sequential([
        # Add the hidden layer with 64 neurons and ReLU (Rectified Linear Unit) activation.
        # 'input_dim' is the size of the input layer, matching the number of features in the input data.
        # This is the first and only hidden layer in this architecture.
        tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),  # Hidden Layer with 64 neurons

        # Add the output layer with a single neuron since this is a binary classification task.
        # The sigmoid activation function is used to ensure the output is between 0 and 1,
        # which can be interpreted as the probability of the positive class.
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output Layer with 1 neuron for binary classification
    ])

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return model


def build_model_3(input_dim):
    """
    Builds a multi-layer perceptron (MLP) model with two hidden layers.

    Parameters:
    ----------
    input_dim: int
        The number of features in the input data. This is used to set the size
        of the input layer, which will accept data for each neuron.

    Returns:
    ---------
    model : Sequential
        The compiled TensorFlow Keras MLP model with the specified architecture.
    """
    # Initialize the model as a linear stack of layers.
    model = tf.keras.models.Sequential([
        # Add the first hidden layer with 128 neurons. The 'relu' activation function is used.
        # 'input_dim' specifies the number of input neurons to the model equal to the number of features.
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),  # Hidden Layer 1 with 128 neurons

        # Add the second hidden layer with 64 neurons, also using the 'relu' activation function.
        tf.keras.layers.Dense(64, activation='relu'),  # Hidden Layer 2 with 64 neurons

        # Add the output layer with a single neuron since it's a binary classification.
        # 'sigmoid' activation is used to output a value between 0 and 1.
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output Layer with 1 neuron
    ])

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return model
