import time

import numpy as np
import inspect
import matplotlib.pyplot as plt
import h5py

from utils.logger import print_status


# Erstellung von Dataset, die aus train und test data besteht.
def create_data(real_images, fake_images, other_images):
    dataset_creator = DatasetCreator(real_images, fake_images, other_images)
    train_data, test_data = dataset_creator.get_train_data(), dataset_creator.get_test_data()
    print_status()
    return train_data, test_data


# Speicherung der Daten
def save_data(real_images, fake_images, other_images):
    with h5py.File('images.h5', 'w') as hf:
        hf.create_dataset('real_images', data=real_images)
        hf.create_dataset('fake_images', data=fake_images)
        hf.create_dataset('other_images', data=other_images)

    print_status()


class DatasetCreator():
    def __init__(self, real_images, fake_images, other_images):
        # Erstellen von Labels
        real_labels = np.ones(len(real_images))
        fake_labels = np.zeros(len(fake_images))
        other_labels = np.full(len(other_images), 2)

        # (schneller) Zusammenführen der Bilder und Labels vor dem Aufteilen
        total_length = len(real_images) + len(fake_images) + len(other_images)
        all_images = np.empty((total_length,) + real_images[0].shape, dtype=real_images[0].dtype)
        # Hinzufügen der realen, gefälschten und anderen Bilder
        all_images[:len(real_images)] = real_images
        all_images[len(real_images):len(real_images) + len(fake_images)] = fake_images
        all_images[len(real_images) + len(fake_images):] = other_images

        # Zusammenführen der Labels
        all_labels = np.concatenate([real_labels, fake_labels, other_labels], axis=0)

        # (Langsamer) Zusammenführen der Bilder und Labels vor dem Aufteilen
        # all_images = np.concatenate([real_images, fake_images], axis=0)
        # all_labels = np.concatenate([real_labels, fake_labels], axis=0)

        # Anwenden der Funktion zum Erstellen von Trainings- und Testsets
        x_train, x_test, y_train, y_test = self.create_training_and_testing_sets(
            all_images, all_labels, train_size=0.8
        )

        self.train_data = (x_train, y_train)
        self.test_data = (x_test, y_test)
        print_status()



    def get_train_data(self):
        print_status()
        return self.train_data

    def get_test_data(self):
        print_status()
        return self.test_data

    def create_training_and_testing_sets(self, x, y, train_size=0.8):
        # print(f"Status in {__file__}, Zeile {inspect.currentframe().f_lineno}: Okay")
        # Bestimmung der Trainings- und Testindizes
        num_train = int(len(x) * train_size)
        indices = np.arange(len(x))
        np.random.shuffle(indices)

        train_indices = indices[:num_train]
        test_indices = indices[num_train:]

        # print(f"Status in {__file__}, Zeile {inspect.currentframe().f_lineno}: Okay")
        # Teilen der Daten in Trainings- und Testsets
        x_train = [x[i] for i in train_indices]
        y_train = [y[i] for i in train_indices]
        x_test = [x[i] for i in test_indices]
        y_test = [y[i] for i in test_indices]

        # print(f"Trainingsdaten: {len(x_train)} Bilder, Testdaten: {len(x_test)} Bilder")
        # print(f"Labels in Trainingsdaten: {np.unique(y_train, return_counts=True)}")
        # print(f"Labels in Testdaten: {np.unique(y_test, return_counts=True)}")
        print_status()
        return x_train, x_test, y_train, y_test

    def test_data_generator(self, data, num_samples=5):
        """ Zeigt eine bestimmte Anzahl von Bildern aus den Trainingsdaten an. """
        x_train, y_train = data

        # Sicherstellen, dass die Anzahl der Samples nicht größer ist als die Menge der verfügbaren Daten
        num_samples = min(num_samples, len(x_train))

        for i in range(num_samples):
            plt.imshow(x_train[i])
            plt.title(f'Label: {y_train[i]}')
            plt.show()
        print_status()
