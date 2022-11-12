import tensorflow as tflow
import tensorflow.python.compat as tf
from keras.applications import vgg16
from keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import callbacks
import cv2
import os
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from datetime import datetime
now = datetime.now()
 



labels = ['for', 'against']
img_size = 224

def get_data(data_dir):
    data = [] 
    for label in labels: 
      
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
             
            except Exception as e:
                print(e)
    return np.array(data)

#train = get_data('data/Smoking/training')
#val = get_data('data/Smoking/testing')

train = get_data('train')
val = get_data('test')

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 223
x_val = np.array(x_val) / 223

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)




datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)





vgg16_model = vgg16.VGG16()

model = Sequential()    #convert to sequential layer, do not include output layer
i = 0
for layer in vgg16_model.layers[:-1]:
    i = i+1
    model.add(layer)

    if ((i % 3) == 0):
        model.add(Dropout(0.4))
    

vgg16_model.summary()


for layer in model.layers:
    layer.trainable = False

model.add(Dense(units=2, activation='softmax'))     #Own outputlayer, trainable

model.summary()


model.compile(optimizer = 'Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 20, 
                                        restore_best_weights = True)


K.set_value(model.optimizer.learning_rate, 0.001)
history = model.fit(x_train,y_train, epochs = 100 , validation_data = (x_val, y_val), callbacks=earlystopping)
# bagging
#vision transformer
#outlier dection
#pruning

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(history.epoch))

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig("vgg16_1.0_"+ str(now.strftime("%d-%m-%Y_%H:%M:%S")) + ".jpg")
plt.show()
model.save('vgg16_1.0')


