from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, test_generator, steps):
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
    cr = classification_report(y_true, y_pred, target_names=['fake', 'real', 'other'])

    # Ausgabe der Confusion Matrix und des Klassifizierungsberichts
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", cr)


