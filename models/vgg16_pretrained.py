from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input


def train_pretrained_vgg16(normalized_train_data, normalized_test_data, train_generator, test_generator, batch_size=32,
                           epochs=10):
    vgg16 = VGG16(
        include_top=False,
        weights='imagenet',
        input_tensor=Input(shape=(224, 224, 3))
    )
    model = Sequential([
        vgg16,
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    history_vgg16_pretrained = model.fit(
        train_generator,
        steps_per_epoch=len(normalized_train_data[0]) // batch_size,
        validation_data=test_generator,
        validation_steps=len(normalized_test_data[0]) // batch_size,
        epochs=epochs
    )
    return history_vgg16_pretrained, model
