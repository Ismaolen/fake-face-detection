import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Deaktiviert GPU
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Pfad zum trainierten Modell und YOLO-Konfigurationen
MODEL_PATH = '../../docs/saved_models/b7e2d344-ce54-4ee3-b6c4-db122ef5d5a8.keras'
YOLO_CONFIG = '../../../yolo/darknet/cfg/yolov3.cfg'
YOLO_WEIGHTS = '../../../yolo/darknet/yolov3.weights'
YOLO_CLASSES = '../../../yolo/darknet/data/coco.names'

# Modell und YOLO laden
model = load_model(MODEL_PATH)
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
with open(YOLO_CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def prepare_image_from_frame(frame):
    """
    Prepares an image for prediction by resizing and normalizing.

    Parameters
    ----------
    frame : np.ndarray
        The input frame to be prepared.

    Returns
    -------
    np.ndarray
        The processed frame ready for prediction, with dimensions expanded for model input.
    """
    img = cv2.resize(frame, (224, 224))  # Skalierung
    img = img.astype("float32") / 255.0  # Normalisierung
    img = np.expand_dims(img, axis=0)
    return img

def get_person_boxes(frame, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Detects persons in a given frame using YOLO object detection.

    Parameters
    ----------
    frame : np.ndarray
        The input image frame on which detection is to be performed.
    confidence_threshold : float, optional
        The threshold for filtering weak detections based on confidence scores.
    nms_threshold : float, optional
        The threshold for non-maximum suppression to eliminate redundant overlapping boxes.

    Returns
    -------
    list
        A list of bounding boxes for detected persons. Each bounding box is represented as
        a list of four integers: [x, y, w, h], where (x, y) is the top-left corner, and
        (w, h) are the width and height of the box.
    """
    # Get the dimensions of the frame for later use in calculations
    height, width, _ = frame.shape

    # Create a blob from the input frame to feed to the YOLO network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Feed the blob into the network and get the network's output
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize lists to hold the bounding boxes and their corresponding confidences
    boxes = []
    confidences = []

    # Iterate over each of the detections
    for out in outs:
        for detection in out:
            # Extract the class score
            scores = detection[5:]
            # Use NumPy to find the index of the class with the highest score
            class_id = np.argmax(scores)
            # Confidence of the prediction
            confidence = scores[class_id]

            # If confidence is above threshold and class_id is 0 (Person), calculate box parameters
            if confidence > confidence_threshold and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate top left corner coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Append box and confidence to their respective lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Perform non-maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences. This function returns the indices of the boxes to keep.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    # Extract the bounding boxes of persons based on the indices after NMS
    if len(indices) > 0:
        # Check if indices are a numpy array and flatten it for easier iteration
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        # Gather the final list of person boxes using list comprehension
        person_boxes = [boxes[i] for i in indices]
    else:
        # If no boxes are left after NMS, return an empty list
        person_boxes = []

    return person_boxes



def live_camera_demo():
    """
    Displays live predictions for person detection using a pre-trained model.

    This function captures video from the default camera, identifies person boxes in each frame, and 
    uses a pre-trained model to classify each detected person as 'Fake' or 'Real'. It displays 
    the prediction and confidence level on the frame and draws a bounding box around the detected person.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    cap = cv2.VideoCapture(0)

    # Define the colors
    rectangle_color = (255, 0, 0)  # Blue for the frame
    text_color = (255, 255, 255)  # White for the text
    background_color = (0, 0, 0)  # Black for the text background

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        person_boxes = get_person_boxes(frame)
        for box in person_boxes:
            x, y, w, h = box

            person_image = frame[y:y+h, x:x+w]
            img = prepare_image_from_frame(person_image)
            prediction = model.predict(img)
            predicted_class = 'Fake' if prediction[0][0] > prediction[0][1] else 'Real'
            confidence = np.max(prediction)

            # Display the class, confidence, and actual label
            label = f"Predicted: {predicted_class} ({confidence:.2f})"

            # Prepare the text size and background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y - text_height - baseline - 10), (x + text_width, y), background_color, -1)

            # Add the frame and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), rectangle_color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        cv2.imshow('Live Prediction', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    live_camera_demo()
