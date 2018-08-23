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

from dataset import Dataset, check_balance

batch_size = 128
num_classes = 2
epochs = 10

# load train and test data
ds = Dataset('dataset.json')
ds.norm_amplitude()
window_size = ds.max_pulse_wides()
print("Min wides: ", ds.min_pulse_wides())
print("Max wides: ", ds.max_pulse_wides())
print("Length: ", ds.length())
print("Sparks: ", ds.sparks())
X, Y = ds.split(window_length=window_size, step=5)

# prepare data for learning in keras
for y in Y: y[0] = 0 if y[1] else 1
x_train = np.array(X[:int(9*len(X)/10)]).astype('float32')
y_train = np.array(Y[:int(9*len(Y)/10)]).astype('float32')
x_test  = np.array(X[int(9*len(X)/10):]).astype('float32')
y_test  = np.array(Y[int(9*len(Y)/10):]).astype('float32')
if len(x_train) != len(y_train) or len(x_test) != len(y_test):
	sys.exit('Error: electromagnetic dataset size does not match!')
print("Train size: ", len(y_train))
print("Test size: ", len(y_test))
print("Train balance: ", check_balance(y_train))
print("Test balance: ", check_balance(y_test))

# architecture of fully connected neural network
model = Sequential()
model.add(Dense(2*window_size, activation='relu', input_shape=(window_size,)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# learning proccess
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# test the model on test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
