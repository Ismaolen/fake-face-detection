import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import sys

from config import num_class


def evaluate_model(model, test_generator, steps, unique_id, log_file_path='docs/evaluated/'):
    """
    Evaluates a model using a test generator and logs the results.

    This function runs predictions using the provided model and test generator for a specified number of steps.
    It calculates and logs the confusion matrix and classification report into a text file. The log file is named
    using a unique identifier and is saved in the specified directory.

    Parameters
    ----------
    model : keras Model
        The model to be evaluated.
    test_generator : Generator
        The generator that yields test data and labels.
    steps : int
        The number of steps for which to run the evaluation.
    unique_id : str
        A unique identifier for naming the log file.
    log_file_path : str, optional
        The directory path where the log file will be saved, default is 'docs/evaluated/'.

    Returns
    -------
    None
    """
    filename = f"{log_file_path}{unique_id}.txt"  # Construct the log file name

    # Redirect standard output to a file
    original_stdout = sys.stdout  # Save the original standard output
    with open(filename, 'w') as f:  # Open the log file for writing
        sys.stdout = f  # Set the standard output to the file

        # Initialize lists for actual and predicted labels
        y_true = []  # List for true labels
        y_pred = []  # List for predicted labels

        # Loop through the test generator and collect predictions
        for _ in range(steps):  # Iterate over the test generator for the given number of steps
            imgs, labels = next(test_generator)  # Get the next batch of images and labels
            preds = model.predict(imgs)  # Predict the labels for the images

            # Convert one-hot encodings to label indices
            y_true.extend(np.argmax(labels, axis=1))  # Extend the true labels list
            y_pred.extend(np.argmax(preds, axis=1))  # Extend the predicted labels list

        # Calculate the confusion matrix and classification report
        cm = confusion_matrix(y_true, y_pred)  # Calculate the confusion matrix
        if num_class != 2:  # Check if there are more than two classes
            cr = classification_report(y_true, y_pred, target_names=['fake', 'real', 'other'])  # Get classification report for multiple classes
        else:
            cr = classification_report(y_true, y_pred, target_names=['fake', 'real'])  # Get classification report for binary classes

        # Print the confusion matrix and classification report
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", cr)
    
    # Reset standard output back to the original
    sys.stdout = original_stdout  # Reset standard output to its original setting

    # Final message in the console
    print(f"Evaluation results have been saved in {filename}.")  # Inform the user that evaluation results have been logged
