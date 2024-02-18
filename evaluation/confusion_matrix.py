import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import sys

from config import num_class


def evaluate_model(model, test_generator, steps, unique_id, log_file_path='docs/evaluated/'):
    filename = f"{log_file_path}{unique_id}.txt"

    # Umleiten der Standardausgabe in eine Datei
    original_stdout = sys.stdout
    with open(filename, 'w') as f:
        sys.stdout = f

        # Initialisierung der Listen für tatsächliche und vorhergesagte Labels
        y_true = []
        y_pred = []

        # Durchlaufen des Test-Generators und Sammeln der Vorhersagen
        for _ in range(steps):
            imgs, labels = next(test_generator)
            preds = model.predict(imgs)

            # Konvertieren von One-Hot-Kodierungen zu Label-Indizes
            y_true.extend(np.argmax(labels, axis=1))
            y_pred.extend(np.argmax(preds, axis=1))

        # Berechnen der Confusion Matrix und des Klassifizierungsberichts
        cm = confusion_matrix(y_true, y_pred)
        if num_class != 2:
            cr = classification_report(y_true, y_pred, target_names=['fake', 'real', 'other'])
        else:
            cr = classification_report(y_true, y_pred, target_names=['fake', 'real'])

        # Ausgabe der Confusion Matrix und des Klassifizierungsberichts
        print("Confusion Matrix:\n", cm)
        print("\nClassification Report:\n", cr)
    
    # Standardausgabe zurücksetzen
    sys.stdout = original_stdout

    # Abschließende Nachricht in der Konsole
    print(f"Evaluationsergebnisse wurden in {filename} gespeichert.")
