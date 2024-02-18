from keras.src.layers import MaxPooling2D
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from config import num_class


def train_custom_vgg16(train_generator, test_generator, batch_size=32, epochs=10):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), strides=(2, 2)),
        # Weitere VGG16 typische Convolutional und MaxPooling-Schichten...
        Flatten(),
        Dense(3 if num_class != 2 else 2, activation='softmax')  # Anpassung für drei Klassen
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Berechnen der Schritte pro Epoche und der Validierungsschritte
    steps_per_epoch = max(1, len(train_generator) // batch_size)
    validation_steps = max(1, len(test_generator) // batch_size)

    history_vgg16_custom = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_generator,
        validation_steps=validation_steps,
        epochs=epochs
    )
    return history_vgg16_custom, model


from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


def train_pretrained_xception(train_generator, test_generator, batch_size, epochs):
    # Laden des Xception-Modells, vortrainiert auf ImageNet
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Deaktivieren des Trainings für die Basisschichten
    for layer in xception_base.layers:
        layer.trainable = False

    # Erstellen des Gesamtmodells mit 2-3 Klassen
    model = Sequential([
        xception_base,
        GlobalAveragePooling2D(),
        Dense(3 if num_class != 2 else 2, activation='softmax')  # 3 Neuronen für 3 Klassen
    ])

    model.compile(optimizer=Adam(), loss=('categorical_crossentropy' if num_class != 2 else 'binary_crossentropy'),
                  metrics=['accuracy'])
    steps_per_epoch = max(1, len(train_generator) // batch_size)
    validation_steps = max(1, len(test_generator) // batch_size)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_generator,
        validation_steps=validation_steps,
        epochs=epochs
    )

    # Erhalten der Vorhersagen vom Testgenerator
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images)

    # Umwandeln der Vorhersagen in Klassenindizes
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(test_labels, axis=1)

    # Vergleich der Vorhersagen mit den tatsächlichen Labels
    for i, (pred, actual) in enumerate(zip(predicted_classes, actual_classes)):
        if pred != actual:
            print(f"Fehler bei Bild {i}: Vorhergesagt {pred}, Tatsächlich {actual}")
            # plt.imshow(test_images[i])
            # plt.show()
        if pred == actual:
            print(f"Richtig bei Bild {i}: Vorhergesagt {pred}, Tatsächlich {actual}")
            # plt.imshow(test_images[i])
            # plt.show()

    return history, model

# Beispielaufruf (Trainings- und Testgenerator müssen entsprechend angepasst werden)
# history, model = train_pretrained_xception(train_generator, test_generator)















def train_pretrained_xception_1(train_generator, test_generator, batch_size, epochs):
    # Laden des Xception-Modells, vortrainiert auf ImageNet
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Deaktivieren des Trainings für die Basisschichten
    xception_base.trainable = False
    # for layer in xception_base.layers:
    # layer.trainable = False

    # Erstellen des Gesamtmodells
    model = Sequential([
        xception_base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),  # Zusätzlicher Dense-Layer mit ReLU-Aktivierung
        Dropout(0.5),  # Dropout-Layer zur Reduzierung von Overfitting
        Dense(3 if num_class != 2 else 2, activation='softmax')  # Anpassung für 2 oder 3 Klassen
    ])

    # Anpassung der Verlustfunktion und der Metriken basierend auf der Anzahl der Klassen
    loss = 'categorical_crossentropy' if num_class != 2 else 'binary_crossentropy'
    model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])

    # Berechnen der Schritte pro Epoche und der Validierungsschritte
    steps_per_epoch = max(1, len(train_generator) // batch_size)
    validation_steps = max(1, len(test_generator) // batch_size)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_generator,
        validation_steps=validation_steps,
        epochs=epochs
    )

    # Erhalten der Vorhersagen vom Testgenerator
    test_images, test_labels = next(test_generator)
    predictions = model.predict(test_images)

    # Umwandeln der Vorhersagen in Klassenindizes
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(test_labels, axis=1)

    # Vergleich der Vorhersagen mit den tatsächlichen Labels
    for i, (pred, actual) in enumerate(zip(predicted_classes, actual_classes)):
        if pred != actual:
            print(f"Fehler bei Bild {i}: Vorhergesagt {pred}, Tatsächlich {actual}")
            # plt.imshow(test_images[i])
            # plt.show()
        if pred == actual:
            print(f"Richtig bei Bild {i}: Vorhergesagt {pred}, Tatsächlich {actual}")
            # plt.imshow(test_images[i])
            # plt.show()

    return history, model
