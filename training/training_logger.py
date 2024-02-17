import datetime
import io
import uuid

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from config import *
from visualizations.plot_metrics import plot_metrics


def document_model_information(model, history):
    # Generieren einer einzigartigen ID und aktuellem Datum
    unique_id = uuid.uuid4()
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    append_model_summary_to_file(model, unique_id, current_datetime)
    append_data_augmentation_to_file(unique_id, current_datetime)
    append_hyperparameters_to_file(unique_id, current_datetime)
    append_training_metrics_from_history(unique_id, current_datetime, history)
    plot_metrics(unique_id, current_datetime, history, path="docs/loss_vs_accuracy_diagrams/")
    save_model_as_h5_file(model, unique_id, current_datetime, "docs/saved_models/")


def save_model_as_h5_file(model, unique_id, current_datetime, file_path="docs/saved_models/"):
    filename = f"{file_path}{unique_id}_{current_datetime}.h5"
    model.save(filename)


def append_model_summary_to_file(model, unique_id, current_datetime,
                                 file_path='docs/model_architecture.txt'):
    # Umleiten der summary-Ausgabe in eine Variable
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()

    # Aktuelles Datum und Uhrzeit

    # Trennzeichen
    separator = "=" * 65

    # Zusammenführen der Ausgabe
    output = (f"\n\n{separator}\n"
              f"ID: {unique_id}\n{current_datetime}\n{separator}\n{summary_string}")

    # Anhängen der Daten an die .txt Datei
    with open(file_path, 'a') as file:
        file.write(output)


def append_data_augmentation_to_file(unique_id, current_date, file_path='docs/data_augmentation.csv'):
    # Daten vorbereiten
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

    # Zeile für die Datei erstellen
    line_to_append = ','.join(str(data_to_append[key]) for key in data_to_append) + '\n'

    # Anhängen der Daten an die Datei
    with open(file_path, 'a') as file:
        file.write(line_to_append)


def append_hyperparameters_to_file(unique_id, current_date, file_path='docs/hyperparameters.csv'):
    # Daten vorbereiten
    data_to_append = {
        'ID': unique_id,
        'Date': current_date,
        'Batch Size': batch_size,
        'Epochs': epochs,
        'Learning Rate': learning_rate,
        'Input Resolution': input_resolution
    }

    # Zeile für die Datei erstellen
    line_to_append = ','.join(str(data_to_append[key]) for key in data_to_append) + '\n'

    # Anhängen der Daten an die Datei
    with open(file_path, 'a') as file:
        file.write(line_to_append)


def append_training_metrics_from_history(unique_id, current_date, history, file_path='docs/training_metrics.csv'):
    epochs = len(history.history['loss'])

    with open(file_path, 'a') as file:
        for epoch in range(epochs):
            train_loss = history.history['loss'][epoch]
            train_accuracy = history.history['accuracy'][epoch]
            val_loss = history.history['val_loss'][epoch]
            val_accuracy = history.history['val_accuracy'][epoch]

            data_to_append = {
                'ID': unique_id,
                'Date': current_date,
                'Train Loss': train_loss,
                'Train Accuracy': train_accuracy,
                'Validation Loss': val_loss,
                'Validation Accuracy': val_accuracy
            }

            line_to_append = ','.join(str(data_to_append[key]) for key in data_to_append) + '\n'
            file.write(line_to_append)
