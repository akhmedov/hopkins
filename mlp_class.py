#
#  mlp_binary_classification.py
#  Hopkins
#
#  Created by Rolan Akhmedov on 22.09.18.
#  Copyright © 2018 Rolan Akhmedov. All rights reserved.
#

from __future__ import print_function

import sys, keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from dataset import Dataset, plot
from dataset import check_balance, make_balance

# load train and test data
ds = Dataset('train.json')
ds.norm_amplitude()
window_size = ds.max_pulse_wides()
X, Y = ds.split(window_length=window_size, step=1)

# prepare data for learning in keras
for y in Y: y[0] = 0 if sum(y) > 1 else 1
x_train = np.array(X[:int(9*len(X)/10)]).astype('float32')
y_train = np.array(Y[:int(9*len(Y)/10)]).astype('float32')
x_test  = np.array(X[int(9*len(X)/10):]).astype('float32')
y_test  = np.array(Y[int(9*len(Y)/10):]).astype('float32')
# x_train, y_train = make_balance(x_train, y_train)
if len(x_train) != len(y_train) or len(x_test) != len(y_test):
	sys.exit('Error: electromagnetic dataset size does not match!')

# print dataset info
print('_________________________________________________________________')
print('Pulse wides (sempales): {} - min, {} - max'.format(ds.min_pulse_wides(), ds.max_pulse_wides()))
print('Size: {} - number of sempales, {} - number of sparks'.format(ds.length(), ds.sparks()))
print('Train: {} - size, {} - balance'.format(len(y_train), check_balance(y_train)))
print('Test: {} - size, {} - balance'.format(len(y_test), check_balance(y_test)))
print('_________________________________________________________________')

# architecture of fully connected neural network
model = Sequential()
model.add(Dense(int(window_size/2), activation='relu', input_shape=(window_size,)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# learning proccess
history = model.fit(x_train, y_train,
					batch_size=256, epochs=10, verbose=1,
					validation_data=(x_test, y_test))

# test the model on test data
model.summary()
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
