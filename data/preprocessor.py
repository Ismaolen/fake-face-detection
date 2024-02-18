import cv2
import numpy as np

from utils.logger import print_status
from config import num_class
from tensorflow.keras.utils import to_categorical


def scale_images(data, target_size=(224, 224)):
    """ Skaliert Bilder auf eine gegebene Größe. """
    scaled_data = []
    for image in data:
        img = cv2.resize(image, target_size)
        scaled_data.append(img)
    print_status()
    return np.array(scaled_data)


def normalize_images(images):
    """ Normalisiert die Bilder auf einen Wertebereich von 0 bis 1. """
    print_status()
    return images / 255.0


def preprocess_data(train_images, test_images, train_labels, test_labels):
    # Scale images: Skaliert die Bilder auf einer Größe von 224X224
    scaled_train_images = scale_images(train_images, (224, 224))
    scaled_test_images = scale_images(test_images, (224, 224))

    # Normalisieren der Bilder
    normalized_train_images = normalize_images(scaled_train_images)
    normalized_test_images = normalize_images(scaled_test_images)

    # Umwandeln in One-Hot-Kodierung
    train_labels = to_categorical(train_labels, num_classes=(3 if num_class != 2 else 2))
    test_labels = to_categorical(test_labels, num_classes=(3 if num_class != 2 else 2))

    # Aktualisierte Trainings- und Testdaten
    normalized_train_data = (normalized_train_images, train_labels)
    normalized_test_data = (normalized_test_images, test_labels)
    print_status()
    return normalized_train_data, normalized_test_data
