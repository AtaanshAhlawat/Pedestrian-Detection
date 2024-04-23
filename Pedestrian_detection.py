import numpy as np
import cv2
import imutils

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

def pedestrian_detection(image, model, layer_names, personidz=0):
    (H, W) = image.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_names)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

    if len(idzs) > 0:
        for i in idzs.flatten():
            (x, y, w, h) = boxes[i]
            confidence = confidences[i]
            centroid = centroids[i]
            results.append((confidence, (x, y, x + w, y + h), centroid))

    return results

# Path to label file and model configurations
labelsPath = "coco.names"
weightsPath = "yolov4-tiny.weights"
configPath = "yolov4-tiny.cfg"

# Load labels from file
try:
    with open(labelsPath, 'r') as f:
        LABELS = f.read().strip().split("\n")
except FileNotFoundError:
    print(f"Error: '{labelsPath}' not found.")
    exit()

# Load YOLOv4-tiny model
model = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Get output layer names
layer_names = model.getUnconnectedOutLayersNames()

# Open video capture
with cv2.VideoCapture("streetup.mp4") as cap:
    while cap.isOpened():
        grabbed, frame = cap.read()
        if not grabbed:
            break

        # Resize frame
        frame = imutils.resize(frame, width=700)

        # Perform pedestrian detection
        results = pedestrian_detection(frame, model, layer_names, personidz=LABELS.index("person"))

        # Display detection results
        for confidence, bbox, centroid in results:
            (startX, startY, endX, endY) = bbox
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        cv2.imshow("Detection", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' key to exit
            break

cv2.destroyAllWindows()
