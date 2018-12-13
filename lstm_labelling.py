#
#  lstm_binary_labelling.py
#  Hopkins
#
#  Created by Rolan Akhmedov on 05.07.18.
#  Copyright © 2018 Rolan Akhmedov. All rights reserved.
#

import gc
import numpy as np

from dataset import Dataset, plot
from dataset import check_balance, make_balance

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SpatialDropout1D
from keras.layers import TimeDistributed

BATCH_SIZE = 1

ds = Dataset('test.json')
ds.norm_amplitude()
window_size = ds.max_pulse_wides()
X, Y = ds.split(window_length=window_size, step=1)

for y in Y: y[0] = 0 if sum(y) > 1 else 1
x_train = np.array(X[:int(9*len(X)/10)]).astype('float32')
y_train = np.array(Y[:int(9*len(Y)/10)]).astype('float32')
x_test  = np.array(X[int(9*len(X)/10):]).astype('float32')
y_test  = np.array(Y[int(9*len(Y)/10):]).astype('float32')
x_test, y_test = make_balance(x_test, y_test)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print('_________________________________________________________________')
print('Pulse wides (sempales): {} - min, {} - max'.format(ds.min_pulse_wides(), ds.max_pulse_wides()))
print('Size: {} - number of sempales, {} - number of sparks'.format(ds.length(), ds.sparks()))
print('Train: {} - size, {} - balance'.format(len(y_train), check_balance(y_train)))
print('Test: {} - size, {} - balance'.format(len(y_test), check_balance(y_test)))
print('_________________________________________________________________')
gc.collect()

model = Sequential()
model.add(LSTM(window_size, batch_input_shape=(BATCH_SIZE, window_size, 1), stateful=True))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(x_train, y_train,
					batch_size=BATCH_SIZE, epochs=1, verbose=1,
					validation_data=(x_test, y_test))

model.summary()
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
