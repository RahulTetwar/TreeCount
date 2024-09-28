from deepforest import main
import matplotlib.pyplot as plt
model = main.deepforest()
model.use_release()
img = model.predict_image(path="pa6.jpg",return_plot=True)
plt.imshow(img[:,:,::-1])
plt.show()




