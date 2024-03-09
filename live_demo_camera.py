import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Pfad zum trainierten Modell
MODEL_PATH = 'docs/saved_models/b7e2d344-ce54-4ee3-b6c4-db122ef5d5a8.keras'

# Modell laden
model = load_model(MODEL_PATH)


def prepare_image_from_frame(frame):
    """Bild für die Vorhersage vorbereiten."""
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def live_camera_demo():
    """Live-Demo, die Echtzeit-Vorhersagen mit der Kamera durchführt."""
    cap = cv2.VideoCapture(0)  # Geräteindex der Kamera (normalerweise 0 für die Hauptkamera)

    class_labels = ['fake', 'real']  # Passen Sie die Labels an Ihr Modell an

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = prepare_image_from_frame(frame)
        prediction = model.predict(img)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        label = f"{predicted_class}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Live Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_camera_demo()
