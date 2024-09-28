import cv2
from deepforest import main

# Define the path to your image
img_path = "abc.jpg"

# Load the DeepForest model with the pre-trained weights
model = main.deepforest("NEON.pt")

# Read the image using OpenCV
img = cv2.imread(img_path)

# Perform object detection and get bounding box information
predictions = model.predict_image(img_path)

# Filter predictions to count trees (you may need to adjust thresholds)
tree_count = len([p for p in predictions if p["label"] == "tree"])

# Display the image and the tree count
cv2.imshow("input", img)
print("Number of trees:", tree_count)
cv2.waitKey(0)
