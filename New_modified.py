import torch
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:latest', 'yolov5s')

# Load the image you want to make predictions on
image_path = "033.jpg"

# Perform object detection
results = model(image_path)

# Display the image with bounding boxes
img = Image.open(image_path)
plt.figure(figsize=(10, 8))
plt.imshow(img)

# Loop through the detected objects and draw rectangles
for detection in results.pred[0]:
    bbox = detection[:4].cpu().numpy()
    label = model.names[int(detection[5])]
    score = detection[4].cpu().item()
    
    if score > 0.5:  # You can adjust the confidence threshold
        x, y, w, h = bbox
        plt.rectangle((x, y), (x + w, y + h), fill=False, edgecolor='red', linewidth=2)
        plt.text(x, y - 5, f"{label} {score:.2f}", color='red')

plt.axis('off')
plt.show()
