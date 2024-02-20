from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

import numpy as np
from config import num_class
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import Xception
from tensorflow.keras.regularizers import l2


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


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def train_pretrained_xception_2(train_generator, test_generator, batch_size, epochs):
    # Laden des Xception-Modells, vortrainiert auf ImageNet

    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Feintuning: Einige der oberen Schichten des Xception-Modells freigeben
    for layer in xception_base.layers[:-5]:
        layer.trainable = False
    for layer in xception_base.layers[-5:]:
        layer.trainable = True
    # xception_base.trainable = False

    # Erstellen des Gesamtmodells
    model = Sequential([
        xception_base,
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),  # Zusätzlicher Dense-Layer
        Dropout(0.5),  # Dropout-Layer
        Dense(3 if num_class != 2 else 2, activation='softmax')  # Anpassung für die Anzahl der Klassen
    ])

    # Optimierer: RMSprop als Alternative zu Adam
    use_rmsprop = True
    optimizer = RMSprop(learning_rate=0.001) if use_rmsprop else Adam(learning_rate=0.001)

    # Modell kompilieren
    # loss = 'categorical_crossentropy' if num_class != 2 else 'binary_crossentropy'
    # mit binary_crossentropy wenn zwei klassen und Daten nicht in one hot-kodierung vorlegen.
    loss = 'categorical_crossentropy'  # da sowieso Daten in one-hot kodiert sind.
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Learning Rate Scheduler einrichten
    lr_scheduler = LearningRateScheduler(scheduler)

    # Modell trainieren
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_generator) // batch_size),
        validation_data=test_generator,
        validation_steps=max(1, len(test_generator) // batch_size),
        epochs=epochs,
        callbacks=[lr_scheduler]  # Learning Rate Scheduler als Callback hinzufügen
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


def train_pretrained_xception_3(train_generator, test_generator, batch_size, epochs):
    # Laden des Xception-Modells, vortrainiert auf ImageNet
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Feintuning: Einige der oberen Schichten des Xception-Modells freigeben
    # Freeze the un-trainable layers of the model base
    for layer in xception_base.layers[:(len(xception_base.layers) - 19)]:
        layer.trainable = False
    for layer in xception_base.layers[(len(xception_base.layers) - 19):]:
        layer.trainable = True
    # xception_base.trainable = False

    # Erstellen des Gesamtmodells
    model = Sequential([
        xception_base,
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),  # L2-Regularisierung
        Dropout(0.5),
        BatchNormalization(),
        Dense(3 if num_class != 2 else 2, activation='softmax')
    ])

    # Optimierer: RMSprop als Alternative zu Adam
    use_rmsprop = True
    optimizer = RMSprop(learning_rate=0.0001) if use_rmsprop else Adam(learning_rate=0.0001)

    # Modell kompilieren
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Learning Rate Scheduler einrichten
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)

    # Modell trainieren
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_generator) // batch_size),
        validation_data=test_generator,
        validation_steps=max(1, len(test_generator) // batch_size),
        epochs=epochs,
        callbacks=[lr_scheduler]  # Learning Rate Scheduler als Callback hinzufügen
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


def train_pretrained_xception_4(train_generator, test_generator, batch_size, epochs):
    # Laden des Xception-Modells, vortrainiert auf ImageNet
    xception_base = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Feintuning: Einige der oberen Schichten des Xception-Modells freigeben
    for layer in xception_base.layers[:-19]:
        layer.trainable = False
    for layer in xception_base.layers[-19:]:
        layer.trainable = True

    # Erstellen des Gesamtmodells
    model = Sequential([
        xception_base,
        BatchNormalization(),
        GlobalAveragePooling2D(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),  # Mehr Neuronen und stärkere Regularisierung
        Dropout(0.7),  # Erhöhte Dropout-Rate
        BatchNormalization(),
        Dense(num_class, activation='softmax')  # Anpassung für die Anzahl der Klassen
    ])

    # Optimierer: RMSprop als Alternative zu Adam
    optimizer = RMSprop(learning_rate=0.00001)

    # Modell kompilieren
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Learning Rate Scheduler und Early Stopping einrichten
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # Modell trainieren
    history = model.fit(
        train_generator,
        steps_per_epoch=max(1, len(train_generator) // batch_size),
        validation_data=test_generator,
        validation_steps=max(1, len(test_generator) // batch_size),
        epochs=epochs,
        callbacks=[lr_scheduler, early_stopping]  # Callbacks hinzufügen
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
