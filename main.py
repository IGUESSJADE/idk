import cv2
import numpy as np

# Load YOLO model and class names
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Set up video capture
cap = cv2.VideoCapture('vid.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for detection`
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getUnconnectedOutLayersNames()

    detections = net.forward(layer_names)

    # Create lists to store the detected person's information
    person_boxes = []
    person_confidences = []

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == 0 and confidence > 0.1:
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                width = int(obj[2] * frame.shape[1])
                height = int(obj[3] * frame.shape[0])

                # Calculate box coordinates
                x = max(0, int(center_x - (width / 2)))
                y = max(0, int(center_y - (height / 2)))

                person_boxes.append((x, y, x + width, y + height))
                person_confidences.append(float(confidence))

    # Apply Non-Maximum Suppression (NMS) to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(person_boxes, person_confidences, 0.5, 0.4)

    # Draw bounding boxes and labels for the detected persons
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, x_plus_w, y_plus_h = person_boxes[i]
            confidence = person_confidences[i]
            cv2.rectangle(frame, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 2)
            label = f"Person: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()