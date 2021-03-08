import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.layers.core import Dropout

train_datagen = image.ImageDataGenerator(
    rescale=1/255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
train_data = train_datagen.flow_from_directory(
    './train', target_size=(256, 256), batch_size=32, class_mode='binary')

test_datagen = image.ImageDataGenerator(rescale=1/255)
test_data = test_datagen.flow_from_directory(
    './test', target_size=(256, 256), batch_size=32, class_mode='binary')

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3,
          activation='relu', input_shape=[256, 256, 3]))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, validation_data=test_data, epochs=25)


def predict_img(img_path):
    path = img_path
    img = image.load(path, target_size=(256, 256))
    img = image.img_to_array(img)/255
    img = np.array(img)
    pred = model.predict(img)
    return pred
