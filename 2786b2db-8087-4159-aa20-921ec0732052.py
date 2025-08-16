import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Load class names (garbage-related)
with open("garbage.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Colors for each class
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Function to get output layer names
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load image
image = cv2.imread("./test_garbage.jpg")
Height, Width = image.shape[:2]
scale = 0.00392

# Prepare input
blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(get_output_layers(net))

# Parameters
conf_threshold = 0.5
nms_threshold = 0.4

boxes, confidences, class_ids = [], [], []

# Process detections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Apply Non-Maximum Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw detections
for i in indexes.flatten():
    x, y, w, h = boxes[i]
    label = str(classes[class_ids[i]])
    confi = round(confidences[i], 2)
    color = COLORS[class_ids[i]]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, f"{label} {confi}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Show result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
