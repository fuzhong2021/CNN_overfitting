import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D, Flatten , Dropout, LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.regularizers import l2
from keras import callbacks
import tensorflow as tf
import cv2
import os
from datetime import datetime
import numpy as np

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
x_train = np.array(x_train) / 224
x_val = np.array(x_val) / 224

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




model = Sequential()
model.add(Conv2D(32,3,padding="same", input_shape=(224,224,3)))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D())
model.add(Dropout(0.4))


model.add(Conv2D(64, 3, padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Conv2D(128, 3, padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Conv2D(256, 3, padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Conv2D(512, 3, padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Conv2D(512, 3, padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Conv2D(512, 3, padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(2, activation="softmax"))

model.summary()

#mean-square didnt wen't well
model.compile(optimizer = 'Adam' , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 100, 
                                        restore_best_weights = True)

K.set_value(model.optimizer.learning_rate, 0.001)
history = model.fit(x_train,y_train,epochs = 100 , validation_data = (x_val, y_val), callbacks=earlystopping)

#model.predict(test_Data)
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
plt.savefig("leaky_relu_"+ str(now.strftime("%d-%m-%Y_%H:%M:%S")) + ".jpg")
plt.show()
model.save('leaky_relu_'+ str(now.strftime("%d-%m-%Y_%H:%M:%S")))


#bagging -> 5 different architectures with a trained model
