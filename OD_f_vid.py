import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()


# Load gender classification model
gender_net = cv2.dnn.readNetFromCaffe("gender.prototxt", "gender.caffemodel",)
gender_list = ['Male', 'Female']

# Initialize variables to store previous frame's bounding boxes and genders
prev_bboxes = []
prev_genders = []

# Open video file or capture device
url = "https://192.168.0.100:8080/video"
cap = cv2.VideoCapture("peoplecount1.mp4")
#cap = cv2.VideoCapture(url)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1020,500))
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    bboxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                bboxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to remove redundant bounding boxes
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)

    # Ensure indices is converted to a numpy array before calling flatten()
    if isinstance(indices, tuple):
        indices = np.array(indices)

    # Detect gender for each person detected
    for i in indices.flatten():
        x, y, w, h = bboxes[i]
        blob = cv2.dnn.blobFromImage(frame, 1, (224, 224), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        prev_genders.append(gender)

        # Draw the bounding boxes and gender labels
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}, Gender: {gender}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    prev_bboxes = bboxes  # Update previous frame's bounding boxes

    # Show the frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
