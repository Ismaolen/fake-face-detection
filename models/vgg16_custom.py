from config import num_class
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


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
