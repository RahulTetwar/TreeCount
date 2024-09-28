from deepforest import main
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Create a DeepForest model instance
model = main.deepforest()

# Specify the model release to use
model.use_release()

# Load an example image (replace "new.png" with your image path)
sample_image_path = "new.png"
img = Image.open(sample_image_path)
img = np.array(img).astype(np.float32)

# Predict objects in the image
predictions = model.predict_image(img, return_plot=False)

# Get the bounding box coordinates and labels for the identified objects
bbox_list = predictions["bounding_box"]
labels = predictions["label"]

# Display the image
plt.imshow(img)

# Plot bounding boxes and count trees
for bbox, label in zip(bbox_list, labels):
    x, y, x2, y2 = bbox
    width = x2 - x
    height = y2 - y
    plt.gca().add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor='r', linewidth=2))
    plt.text(x, y, label, bbox=dict(fill=True, color='white'))

# Count the trees
tree_count = len(bbox_list)
plt.title(f"Detected Trees: {tree_count}")

# Show the image with bounding boxes
plt.show()

print(f"Total Trees Detected: {tree_count}")
