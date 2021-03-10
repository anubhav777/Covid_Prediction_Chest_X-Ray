import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from glob import glob
from tensorflow.keras.models import Model

IMAGE_SIZE = [224, 224]
train_path = './train'
valid_path = './test'

vgg = VGG16(input_shape=IMAGE_SIZE+[3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

folders = glob('train/*')

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])


train_datagen = image.ImageDataGenerator(
    rescale=1/255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
train_data = train_datagen.flow_from_directory(
    './train', target_size=(256, 256), batch_size=32, class_mode='binary')

test_datagen = image.ImageDataGenerator(rescale=1/255)
test_data = test_datagen.flow_from_directory(
    './test', target_size=(256, 256), batch_size=32, class_mode='binary')

model.fit_generator(train_data, validation_data=test_data, epochs=25)


def predict_img(img_path):
    path = img_path
    img = image.load(path, target_size=(256, 256))
    img = image.img_to_array(img)/255
    img = np.array(img)
    pred = model.predict(img)
    return pred
