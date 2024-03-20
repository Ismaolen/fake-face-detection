import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input


def train_pretrained_vgg16_weighted(normalized_train_data, normalized_test_data, train_generator, test_generator,
                                    train_labels, batch_size=32, epochs=10):
    weights = compute_class_weights(train_labels)
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
    history_vgg16_weighted_loss = model.fit(
        train_generator,
        steps_per_epoch=len(normalized_train_data[0]) // batch_size,
        validation_data=test_generator,
        validation_steps=len(normalized_test_data[0]) // batch_size,
        epochs=epochs, class_weight=weights
    )
    return history_vgg16_weighted_loss, model


def compute_class_weights(labels):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return {i: weight for i, weight in enumerate(class_weights)}
