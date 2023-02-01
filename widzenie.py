from tensorflow import keras
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np


test_image = keras.utils.load_img('obraz.png',color_mode="grayscale",target_size=(24, 24))

plt.imshow(test_image)
plt.show()


test_image = keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)


#print(test_image.shape)

model = keras.models.load_model("model")

#model.summary()
#plot_model(model, to_file="my_model.png", show_shapes=True)

predictions = model.predict(test_image)
print(predictions)
y_classes = predictions.argmax(axis=-1)
labelmap = ['a', 'b', 'c', 'd', 'e', 'm', 'n', 'ą']
print("Obraz ten klasyfikuje jako: %s" % labelmap[y_classes[0]])