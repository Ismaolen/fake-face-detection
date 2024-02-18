import inspect
import os
import numpy as np
from PIL import Image
import h5py
from config import num_class

from utils.logger import print_status


# Laden der Daten aus bestimmten Pfad
def load_data(path):
    # lädt die Daten aus 'data', die in data_loader gespeichert werden
    global other_images
    data_loader = DataLoader(dataset_path=path)
    # die Funktion des Konstruktors gibt ein Tupel zurück,
    # das zwei NumpyArrays enthält, die in den variablen
    # gespeichert werden
    if num_class == 2:
        real_images, fake_images = data_loader.load_dataset()
        print_status()
        return real_images, fake_images
    else:
        real_images, fake_images, other_images = data_loader.load_dataset()
        print_status()
        return real_images, fake_images, other_images


def load_images():
    # Laden der Bilder und sie der fake und real_images variable zuweisen
    if num_class == 2:
        with h5py.File('images.h5', 'r') as hf:
            real_images = hf['real_images'][:]
            fake_images = hf['fake_images'][:]
        print_status()
        return real_images, fake_images
    else:
        with h5py.File('images.h5', 'r') as hf:
            real_images = hf['real_images'][:]
            fake_images = hf['fake_images'][:]
            other_images = hf['other_images'][:]
        print_status()
        return real_images, fake_images, other_images


class DataLoader:
    def __init__(self, dataset_path):
        print_status()
        self.dataset_path = dataset_path

    def load_dataset(self):

        global other_images, other_images_path
        real_faces_path = os.path.join(self.dataset_path, 'real_faces')
        fake_faces_path = os.path.join(self.dataset_path, 'fake_faces')
        # other_images_path = os.path.join(self.dataset_path, 'apple_red_1')
        if num_class != 2:
            other_images_path = os.path.join(self.dataset_path, 'apple_red_skalled')

        # Laden der Bilder und Labels
        real_images = self.load_images_from_folder(real_faces_path)
        fake_images = self.load_images_from_folder(fake_faces_path)
        if num_class != 2:
            other_images = self.load_images_from_folder(other_images_path)
        print_status()
        if num_class != 2:
            return real_images, fake_images, other_images
        else:
            return real_images, fake_images

    def load_image(self, file_path):
        try:
            with Image.open(file_path) as img:
                return np.array(img)
        except IOError:
            print(f"Kann Bild nicht laden: {file_path}")
            return None

    def load_images_from_folder(self, folder):
        images = []
        for filename in os.listdir(folder):
            img = self.load_image(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
        print_status()
        return np.array(images)
