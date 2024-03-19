import cv2
import os

def extract_label_from_filename(filename):
    # Prüft, ob 'fake' oder 'real' im Dateinamen enthalten ist und gibt das entsprechende Label zurück
    if 'fake' in filename.lower():
        return 'Fake'
    elif 'real' in filename.lower():
        return 'Real'
    else:
        return 'Unbekannt'

def create_diashow(directory, display_time=7000):  # Zeit in Millisekunden
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)

            label = extract_label_from_filename(filename)

            # Ermittelt die Dimensionen des Bildes
            height, width, _ = image.shape

            # Textfarbe und Hintergrundfarbe festlegen
            font_color = (50, 50, 50)  # Dunkelgrau für bessere Sichtbarkeit auf hellem Hintergrund
            background_color = (220, 220, 220)  # Helles Grau als Hintergrund

            # Position des Textes festlegen (hier: unten links)
            text_position = (200, height - 10)  # 10 Pixel vom linken Rand und 20 Pixel vom unteren Rand

            # Textgröße und Dicke
            font_scale = 0.7
            thickness = 2

            # Berechnung der Textgröße für den Hintergrund
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(image, (text_position[0], text_position[1] + baseline), 
                          (text_position[0] + text_width, text_position[1] - text_height), 
                          background_color, -1)

            # Text auf das Bild schreiben
            cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

            cv2.imshow('Diashow', image)
            if cv2.waitKey(display_time) & 0xFF == ord('q'):  # Warten und mit 'q' vorzeitig beenden
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Pfad zum Verzeichnis mit Ihren Bildern
    directory = "demo_images/"
    create_diashow(directory)
