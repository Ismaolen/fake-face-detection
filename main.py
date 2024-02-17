from augmentation.augmenter import create_train_generator, create_test_generator
from config import *
from data.data_loader import load_images, load_data
from data.dataset_creator import create_data, save_data
from data.preprocessor import preprocess_data
from evaluation.confusion_matrix import evaluate_model
from models.vgg16_model import train_custom_vgg16, train_pretrained_xception
from models.vgg16_pretrained import train_pretrained_vgg16
from models.vgg16_pretrained_weighted_loss import train_pretrained_vgg16_weighted
from training.training_logger import document_model_information
import tensorflow as tf


#
def main():
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #    try:
    #        tf.config.set_logical_device_configuration(
    #            gpus[0],
    #            [tf.config.LogicalDeviceConfiguration(memory_limit=12288)])
    #        logical_gpus = tf.config.list_logical_devices('GPU')
    #        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #    except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    #        print(e)

    # print("Verfügbare Geräte: ", tf.config.list_physical_devices())

    # Load Data from the Path
    # real_images, fake_images, other_images = load_data('data/')

    # Save the Fake and Real Images
    # save_data(real_images, fake_images, other_images)

    # Load the Fake and Real Images
    real_images, fake_images, other_images = load_images()

    # Aufteilen in Tests und Trains images
    train_data, test_data = create_data(real_images, fake_images, other_images)

    # Extract Images and Labels
    (train_images, train_labels), (test_images, test_labels) = train_data, test_data

    # Scale, Normalize and Update the Data
    normalized_train_data, normalized_test_data = preprocess_data(train_images, test_images, train_labels, test_labels)

    # Erstellen eines ImageDataGenerator für das Training mit Augmentation
    train_generator = create_train_generator(normalized_train_data)
    # Ein einfacher Generator ohne Augmentation für die Testdaten
    test_generator = create_test_generator(normalized_test_data)

    # Anzeigen einiger augmentierten Trainingsbilder
    # display_augmented_images(train_generator, num_samples=5)

    #history_pretrained_xception, model_pretrained_xception = train_pretrained_xception(train_generator, test_generator,
    #                                                                                   batch_size, epochs)
    history, model = train_custom_vgg16(train_generator, test_generator, batch_size, epochs)
    print(history.history)
    document_model_information(model, history)


    test_steps = len(test_generator)
    # Evaluierung des Modells
    evaluate_model(model, test_generator, test_steps)

    '''
    # Create and train models
    history_vgg16_custom, model_vgg16_custom = (
        train_custom_vgg16(normalized_train_data,
                           normalized_test_data,
                           train_generator,
                           test_generator,
                           batch_size,
                           epochs)
    )
    print(history_vgg16_custom.history)

    # Aufrufen der Funktion
    document_model_information(model_vgg16_custom, history_vgg16_custom)

    history_vgg16_pretrained, model_vgg16_pretrained = (
        train_pretrained_vgg16(normalized_train_data,
                               normalized_test_data,
                               train_generator,
                               test_generator,
                               batch_size,
                               epochs)
    )

    print(history_vgg16_pretrained.history)
    document_model_information(model_vgg16_pretrained, history_vgg16_pretrained)

    history_vgg16_weighted_loss, model_vgg16_weighted_loss = train_pretrained_vgg16_weighted(
        normalized_train_data,
        normalized_test_data, train_generator,
        test_generator, train_labels,
        batch_size, epochs)
    # Ausgabe der Modellzusammenfassungen

    document_model_information(model_vgg16_weighted_loss, history_vgg16_weighted_loss)

    # Auf die History-Attribute zugreifen, um Loss und Genauigkeit zu dokumentieren
    print(history_vgg16_weighted_loss.history)
    '''


if __name__ == "__main__":
    main()
