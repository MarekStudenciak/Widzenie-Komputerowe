from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

model = keras.models.load_model("model")

test_ds = keras.utils.image_dataset_from_directory(
    directory='dane testowe/',
    batch_size=100,
    image_size=(24, 24),
    color_mode='grayscale',
    seed=2137)

print(test_ds.class_names)

for x, y in test_ds:
    X_test = x
    Y_test = y

print(Y_test)

Y_pred = np.argmax(model.predict(X_test), axis=1)

print(Y_pred)

cm = confusion_matrix(Y_test, Y_pred, labels=np.unique(Y_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(Y_test)).plot()
plt.show()