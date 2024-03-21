import datetime
import io
import uuid

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from config import *
from visualizations.plot_metrics import plot_metrics


def document_model_information(model, history):
    """
    Documents various information about a trained model, including architecture, data augmentation parameters,
    hyperparameters, and training metrics. This information is saved in separate files.

    Parameters
    ----------
    model : keras Model
        The trained model whose information is to be documented.
    history : History
        The history object containing the training and validation metrics per epoch.

    Returns
    -------
    uuid.UUID
        The unique identifier assigned to this model and its associated information.
    """
    # Generate a unique identifier and the current date and time
    unique_id = uuid.uuid4()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    # Document various aspects of the model and its training
    append_model_summary_to_file(model, unique_id, current_datetime)  # Append model summary
    append_data_augmentation_to_file(unique_id, current_datetime)  # Append data augmentation parameters
    append_hyperparameters_to_file(unique_id, current_datetime)  # Append hyperparameters
    append_training_metrics_from_history(unique_id, current_datetime, history)  # Append training metrics
    plot_metrics(unique_id, history, path="docs/loss_vs_accuracy_diagrams/")  # Plot and save metrics diagrams
    save_model_as_keras_file(model, unique_id, "docs/saved_models/")  # Save the model to a Keras file
    return unique_id  # Return the unique identifier for reference


def save_model_as_keras_file(model, unique_id, file_path="docs/saved_models/"):
    """
    Saves the trained model as a Keras file (.keras).

    Parameters
    ----------
    model : keras Model
        The trained model to be saved.
    unique_id : uuid.UUID
        The unique identifier for the model.
    file_path : str, optional
        The file path where the model should be saved, default is 'docs/saved_models/'.

    Returns
    -------
    None
    """
    # Construct the filename using the unique ID
    filename = f"{file_path}{unique_id}.keras"
    model.save(filename)  # Save the model as a Keras file


def append_model_summary_to_file(model, unique_id, current_datetime, file_path='docs/model_architecture.txt'):
    """
    Appends the summary of the model to a text file.

    Parameters
    ----------
    model : keras Model
        The model whose summary is to be documented.
    unique_id : uuid.UUID
        The unique identifier for the model.
    current_datetime : str
        The current date and time.
    file_path : str, optional
        The file path of the document to append the summary, default is 'docs/model_architecture.txt'.

    Returns
    -------
    None
    """
    # Redirect the summary output to a StringIO stream
    stream = io.StringIO()
    model.summary(print_fn=lambda x, **kwargs: stream.write(x + '\n'))  # Write model summary to the stream
    summary_string = stream.getvalue()  # Retrieve string from stream
    stream.close()  # Close the stream

    # Formatting the output
    separator = "=" * 65
    output = (f"\n\n{separator}\n"
              f"ID: {unique_id}\n{current_datetime}\n{separator}\n{summary_string}")

    # Append the model summary to the specified file
    with open(file_path, 'a') as file:
        file.write(output)  # Write the formatted summary to the file


def append_data_augmentation_to_file(unique_id, current_date, file_path='docs/data_augmentation.csv'):
    """
    Appends data augmentation parameters to a CSV file.

    Parameters
    ----------
    unique_id : uuid.UUID
        The unique identifier for the model.
    current_date : str
        The current date.
    file_path : str, optional
        The file path of the CSV to append the data augmentation parameters, default is 'docs/data_augmentation.csv'.

    Returns
    -------
    None
    """
    # Prepare the data to be appended
    data_to_append = {
        'ID': unique_id,
        'Date': current_date,
        'Rotation Range': rotation_range,
        'Width Shift Range': width_shift_range,
        'Height Shift Range': height_shift_range,
        'Shear Range': shear_range,
        'Zoom Range': zoom_range,
        'Horizontal Flip': horizontal_flip,
        'Vertical Flip': vertical_flip
    }

    # Create a line for the CSV file
    line_to_append = ','.join(str(data_to_append[key]) for key in data_to_append) + '\n'

    # Write the data augmentation parameters to the CSV file
    with open(file_path, 'a') as file:
        file.write(line_to_append)  # Append the data augmentation parameters to the file


def append_hyperparameters_to_file(unique_id, current_date, file_path='docs/hyperparameters.csv'):
    """
    Appends the model's hyperparameters to a CSV file.

    Parameters
    ----------
    unique_id : uuid.UUID
        The unique identifier for the model.
    current_date : str
        The current date.
    file_path : str, optional
        The file path of the CSV to append the hyperparameters, default is 'docs/hyperparameters.csv'.

    Returns
    -------
    None
    """
    # Prepare the hyperparameters data for appending
    data_to_append = {
        'ID': unique_id,
        'Date': current_date,
        'Batch Size': batch_size,
        'Epochs': epochs,
        'Learning Rate': learning_rate,
        'Input Resolution': input_resolution
    }

    # Create a CSV format line from the hyperparameters
    line_to_append = ','.join(str(data_to_append[key]) for key in data_to_append) + '\n'

    # Append the hyperparameters to the file
    with open(file_path, 'a') as file:
        file.write(line_to_append)  # Write the hyperparameters to the CSV file


def append_training_metrics_from_history(unique_id, current_date, history, file_path='docs/training_metrics.csv'):
    """
    Appends training metrics to a CSV file from the training history.

    Parameters
    ----------
    unique_id : uuid.UUID
        The unique identifier for the model.
    current_date : str
        The current date.
    history : History
        The history object from model training containing metrics per epoch.
    file_path : str, optional
        The file path of the CSV to append the training metrics, default is 'docs/training_metrics.csv'.

    Returns
    -------
    None
    """
    # Calculate the number of epochs from the history
    epochs = len(history.history['loss'])

    # Open the CSV file to append training metrics
    with open(file_path, 'a') as file:
        for epoch in range(epochs):
            # Retrieve training and validation metrics for the current epoch
            train_loss = history.history['loss'][epoch]
            train_accuracy = history.history['accuracy'][epoch]
            val_loss = history.history['val_loss'][epoch]
            val_accuracy = history.history['val_accuracy'][epoch]

            # Prepare the data for appending
            data_to_append = {
                'ID': unique_id,
                'Date': current_date,
                'Train Loss': train_loss,
                'Train Accuracy': train_accuracy,
                'Validation Loss': val_loss,
                'Validation Accuracy': val_accuracy
            }

            # Create a CSV format line from the training metrics
            line_to_append = ','.join(f"{key}:{data_to_append[key]}" for key in data_to_append) + '\n'

            # Append the training metrics for the current epoch to the file
            file.write(line_to_append)
