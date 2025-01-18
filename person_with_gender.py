
from ultralytics import YOLO
import cv2



cap = cv2.VideoCapture('peoplecount1.mp4')
#cap = cv2.VideoCapture('img_1.jpg')

model_path = "best.pt"

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.4

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 500))
    H, W, _ = frame.shape
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)


    # Show the frame
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()