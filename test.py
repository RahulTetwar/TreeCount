from deepforest import main
import cv2
import matplotlib.pyplot as plt

img_path = "new.png"
img = cv2.imread(img_path)

my_model = main.deepforest()
my_model.use_release()

cv2.imshow('input',img)
cv2.waitKey(0)

box_info = my_model.predict_image(img_path, return_plot=False)
box_info


for n in range(len(box_info)):
    x=(box_info.xmin[n]+box_info.xmax[n])/2
    y=(box_info.ymin[n]+box_info.ymax[n])/2
    cv2.circle(img, (int(x), int(y)), 25 , (0, 255 ,0) , 2)

cv2.putText(img,'Total Trees ' + str(len(box_info)), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (255,255,255))

cv2.imshow('Output',img)
cv2.waitKey(0)