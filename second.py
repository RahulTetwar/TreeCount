from deepforest import main
import cv2

img_path = "abc.jpg"
img = cv2.imread(img_path)

# Load the pre-trained deepforest model
my_model = main.deepforest()
my_model.use_release()

cv2.imshow('input', img)
cv2.waitKey(0)

# Perform object detection and get the bounding box information
box_info = my_model.predict_image(img_path, return_plot=True)

for n in range(len(box_info)):
    x = (box_info.xmin[n] + box_info.xmax[n]) // 2
    y = (box_info.ymin[n] + box_info.ymax[n]) // 2
    cv2.circle(img, (int(x), int(y)), 25, (0, 255, 0), 2)

cv2.putText(img, 'Total Trees ' + str(len(box_info)), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

# Display the image with detected objects
cv2.imshow('output', img)
cv2.waitKey(0)
# cv2.destroyAllWindows()
