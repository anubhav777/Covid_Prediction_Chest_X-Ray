import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from glob import glob
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

IMAGE_SIZE = [224, 224]
train_path = './train'
valid_path = './test'

res = ResNet50(input_shape=(224, 224, 3), include_top=False)

for layer in res.layers:
    layer.trainable = False

folders = glob('train/*')

x = Flatten()(res.output)
x = Dense(units=2, activation='sigmoid', name='predictions')(x)

model = Model(res.input, x)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

es = EarlyStopping(monitor="val_accuracy",
                   min_delta=0.01, patience=3, verbose=1)
mc = ModelCheckpoint(filepath="bestmodel.h5",
                     monitor="val_accuracy", verbose=1, save_best_only=True)

train_datagen = image.ImageDataGenerator(
    rescale=1/255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
train_data = train_datagen.flow_from_directory(
    './train', target_size=(244, 244), batch_size=32, class_mode='binary')

test_datagen = image.ImageDataGenerator(rescale=1/255)
test_data = test_datagen.flow_from_directory(
    './test', target_size=(244, 244), batch_size=32, class_mode='binary')

model.fit_generator(train_data, steps_per_epoch=10, epochs=30,
                    validation_data=test_data, validation_steps=16, callbacks=[es, mc])


def predict_img(img_path):
    path = img_path
    img = image.load(path, target_size=(256, 256))
    img = image.img_to_array(img)/255
    img = np.array(img)
    pred = model.predict(img)
    return pred
