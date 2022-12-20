
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
import pandas as pd
dataset = "/content/drive/MyDrive/images"
import cv2
import imghdr
image_exts = ['jpeg','jpg','bmp','png']
import os
for image_class in os.listdir(dataset):
	for image in os.listdir(os.path.join(dataset, image_class)):
		image_path = os.path.join(dataset, image)
		try:
			img = cv2.imread(image_path) 
			tip = imghdr.what(image_path) 
			if tip not in image_exts:
				print('does not exits in list {}'. format(image_path))
				os.remove(image_path)
		except Exception as e:
			print('Issue with image {}'. format(image_path))

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
dataset = tf.keras.utils.image_dataset_from_directory("/content/drive/MyDrive/images")
data_iterator = dataset.as_numpy_iterator()
batch = data_iterator.next()

data = dataset.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

len(data)
train_size = int(len(data)*.5)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1
train_size+val_size+test_size

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test  = data.skip(train_size+val_size).take(test_size)
len(test)


from keras.layers.convolutional import Conv2D
from keras.layers import Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.models import Sequential
model=Sequential()
model.add(Conv2D(64, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
							 metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"),
                       tf.keras.metrics.FalseNegatives()])

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
hist.history

fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()


fig = plt.figure()
plt.plot(hist.history['acc'], color='teal', label='accuracy')
plt.plot(hist.history['val_acc'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

from keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

len(test)
for batch in test.as_numpy_iterator():
	X,y = batch
	yhat = model.predict(X)
	pre.update_state(y,yhat)
	re.update_state(y,yhat)
	acc.update_state(y,yhat)

print(f'Precision:({pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()})')

import  cv2
img =cv2.imread('/content/drive/MyDrive/k1.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()
resize.shape
np.expand_dims(resize, 0).shape
yhat = model.predict(np.expand_dims(resize/255, 0))
yhat
if yhat > 0.5:
	print(f'Predicted class is a Puppy')
else:
	print(f'Predicted class is a Kitten')