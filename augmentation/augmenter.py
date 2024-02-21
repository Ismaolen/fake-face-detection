from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import *
from utils.logger import print_status


def create_train_generator(train_data, batch_size_f=batch_size):
    train_datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        fill_mode=fill_mode,
        # new
        # brightness_range=[0.8, 1.2],
        # channel_shift_range=20.0,
    )
    print_status()
    return train_datagen.flow(train_data[0], train_data[1], batch_size=batch_size_f)


def create_test_generator(test_data, batch_size_f=batch_size):
    test_datagen = ImageDataGenerator()
    # new
    # test_datagen = ImageDataGenerator(rescale=1.0/255.0) wurde schon skaliert, deshalb gelöscht. 
    print_status()
    return test_datagen.flow(test_data[0], test_data[1], batch_size=batch_size_f)


def display_augmented_images(generator, num_samples=5):
    for i in range(num_samples):
        img, label = next(generator)
        plt.imshow(img[0])
        plt.title(f'Label: {label[0]}')
        plt.show()
    print_status()
