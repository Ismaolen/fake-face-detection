import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Pfad zum trainierten Modell
MODEL_PATH = '../../../Data_HTW_Cloud/saved_models/b7e2d344-ce54-4ee3-b6c4-db122ef5d5a8.keras'

# Modell laden
model = load_model(MODEL_PATH)


def prepare_image(image_path):
    """Bild für die Vorhersage vorbereiten."""
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalisierung, falls während des Trainings verwendet
    return img


def live_demo(image_path):
    """Live-Demo, die eine Testbildvorhersage durchführt."""
    # image_path = input("Geben Sie den Pfad zum Bild ein: ")
    img = prepare_image(image_path)
    prediction = model.predict(img)

    # Angenommen, Ihr Modell gibt Wahrscheinlichkeiten für jede Klasse zurück
    class_labels = ['fake', 'real']  # Passen Sie die Labels an Ihr Modell an
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    print(f"Das Bild wird als '{predicted_class}' klassifiziert mit einer Konfidenz von {confidence:.2f}.")
    if predicted_class == 'real':
        return 1
    else:
        return 0


if __name__ == "__main__":
    count_fake = 0
    count_real = 0
    for i in range(15000):
        image_path = f"../../../Data_HTW_Cloud/images/fake_faces/{i + 1}_fake_faces.jpg"
        state = live_demo(image_path)
        if state == 1:
            count_real += 1
        else:
            count_fake += 1

    print(f"Fake count: {count_fake}\n"
          f"Real count: {count_real}\n")
