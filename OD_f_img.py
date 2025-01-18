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
gender_net = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel",)
gender_list = ['Male', 'Female']

# Load image
img = cv2.imread("img_1.jpg")
img = cv2.resize(img,(1080,720))
height, width, channels = img.shape
mean_v = np.mean(img)

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.004, (416, 416), mean_v, True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Initialize lists for bounding boxes, confidences, and class IDs
bboxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.8:
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            bboxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression (NMS) to remove redundant bounding boxes
indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)

# Draw the bounding boxes after applying NMS
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = bboxes[i]
        blob = cv2.dnn.blobFromImage(img, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender_prob = gender_preds[0][0]  # Assuming gender_preds is a 2D array
        gender_label = "Male" if gender_prob > 0.5 else "Female"  # Adjust threshold as needed
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}, Gender: {gender_label}"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


#show the image
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
