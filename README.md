## YOLOv4 Real-Time Object Detection Using Laptop Camera
### Aim:

To perform real-time object detection using a pre-trained YOLOv4 model through a laptop webcam and display detected objects inline in a Jupyter Notebook.

### Algorithm:

1. Initialize the webcam using OpenCV.
2. Load the YOLOv4 pre-trained model (yolov4.weights) and configuration (yolov4.cfg).
3. Load the COCO class labels (coco.names) and assign random colors for visualization.
4. Capture frames continuously from the webcam.
5. Preprocess each frame by creating a blob and pass it through the YOLOv4 network.
6. Extract detected object bounding boxes, class IDs, and confidences.
7. Apply Non-Maximum Suppression (NMS) to remove overlapping boxes.
8. Draw bounding boxes and labels for detected objects on the frame.
9. Display the frames inline in Jupyter Notebook using matplotlib.
10. Stop detection by pressing the “Stop Detection” button in the notebook.

### Program:
```py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
from threading import Thread

# Load YOLOv4
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Load COCO classes
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Get output layers
layer_names = net.getLayerNames()
outs = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in outs.flatten()]

# Start webcam
cap = cv2.VideoCapture(0)

# Create a stop button
stop_button = widgets.Button(description="Stop Detection", button_style='danger')
display(stop_button)
stop_flag = False
def stop_detection(b):
    global stop_flag
    stop_flag = True
stop_button.on_click(stop_detection)

def detect_objects():
    global stop_flag
    while not stop_flag:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape

        # Prepare blob and forward pass
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids, confidences, boxes = [], [], []

        for detection in outs[0]:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indexes.flatten() if len(indexes) > 0 else []:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        clear_output(wait=True)
        plt.imshow(frame_rgb)
        plt.axis('off')
        display(plt.gcf())

thread = Thread(target=detect_objects)
thread.start()
```
### Output


<img width="796" height="604" alt="image" src="https://github.com/user-attachments/assets/80f2f977-f316-4ef9-9b4e-f7273f8858ea" />
### Result:

The webcam captures live video frames.

YOLOv4 detects objects like person, chair, laptop, bottle, etc. in real-time.

Detected objects are highlighted with bounding boxes and class labels.

Frames are displayed inline in the Jupyter Notebook.

Detection stops when the “Stop Detection” button is pressed.



