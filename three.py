import cv2
import numpy as np

# Load YOLO model and configuration files
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names (for YOLO pre-trained model)
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Read the image
image = cv2.imread("abc.jpg")
height, width = image.shape[:2]

# Create a blob from the image (pre-processing)
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set the input to the neural network
net.setInput(blob)

# Run forward pass to get detections
output_layers_names = net.getUnconnectedOutLayersNames()
layer_outputs = net.forward(output_layers_names)

# Initialize variables to count trees
tree_count = 0

# Process each detection
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and classes[class_id] == "tree":
            tree_count += 1

# Print the number of trees detected
print("Number of trees:", tree_count)

# Display the image with bounding boxes (optional)
for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and classes[class_id] == "tree":
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
