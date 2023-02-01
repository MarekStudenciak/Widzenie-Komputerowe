from tensorflow import keras
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout


train_ds, validation_ds = keras.utils.image_dataset_from_directory(
    directory='dane uczące - niezbalansowane/',
    batch_size=100,
    image_size=(24, 24),
    color_mode='grayscale',
    validation_split=0.2,
    subset="both",
    seed=2137)

class_names = train_ds.class_names
print(class_names)
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()


model = keras.Sequential([
    Conv2D(32, kernel_size=(3, 3), input_shape=(24, 24, 1)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, kernel_size=(3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(64),
    Activation('relu'),
    Dropout(0.5),
    Dense(len(class_names)),
    Activation('sigmoid')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit(train_ds, epochs=100, validation_data=validation_ds)
model.save("model")