# -*- coding:utf-8 -*-
import os
import sys
import keras

# system root
sys.path.append('../')

from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from model.simple_cnn import SimpleCNNBuilder
from model.resnet import ResnetBuilder

# load mnist
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# normalize
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# one hot
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)

shape = (32, 32, 3)

# train
epochs = 10
batch_size = 128

# ckpt file path
param_folder = '../param_resnet'
if not os.path.isdir(param_folder):
    os.makedirs(param_folder)

# callback
cbk = ModelCheckpoint(filepath = os.path.join(param_folder, 'param{epoch:02d}.hdf5'))

# train
#builder = SimpleCNNBuilder()
builder = ResnetBuilder()
model = builder.build(input_shape=shape, class_num=10)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
model.fit(x_train, y_train, batch_size, epochs=epochs, verbose=1, callbacks=[cbk], validation_data=(x_test, y_test))

# test
result = model.predict(x_test)
# model 評価
score = model.evaluate(x_test, y_test, verbose=0)

# output
print('test result: ', result.argmax(axis=1))
print('test loss and acc: ' , score)
