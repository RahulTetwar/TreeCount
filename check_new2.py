from deepforest import main
import cv2

# Load the image using OpenCV
img_path = "pa4.jpg"
img = cv2.imread(img_path)

my_model = main.deepforest()
my_model.use_release()

box_info = my_model.predict_image(img, return_plot=False)  # Pass the NumPy array here

for n in range(len(box_info)):
    x = (box_info.xmin[n] + box_info.xmax[n]) / 2
    y = (box_info.ymin[n] + box_info.ymax[n]) / 2
    cv2.circle(img, (int(x), int(y)), 25, (0, 255, 0), 2)
    # cv2.rectangle(img, (int(x), int(y)), 25, (0, 255, 0), 2)

# Change the font to a larger and bolder one (Hershey Complex)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
# font_color = (0, 0, 0)  # Black color
font_color = (255, 255, 255)  # Black color

# Add the text with the specified font properties
cv2.putText(img, 'Total Trees ' + str(len(box_info)), (3, 40), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

cv2.imshow('Output', img)
cv2.waitKey(0)
