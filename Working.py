from deepforest import main
from deepforest import get_data
import os
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
model = main.deepforest()
model.use_release()
# sample_image = get_data("OSBS_029.png")
#img = model.predict_image(path="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB/TEAK_049_2019.tif",return_plot=True)
# img = model.predict_image(path=sample_image,return_plot=True)
img = model.predict_image(path="3.jpg",return_plot=True)

#predict_image returns plot in BlueGreenRed (opencv style), but matplotlib likes RedGreenBlue, switch the channel order.
plt.imshow(img[:,:,::-1])
plt.show()




