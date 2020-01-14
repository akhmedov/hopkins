#
#  121_classs.py
#  Hopkins
#
#  Created by Rolan Akhmedov on 22.09.18.
#  Copyright © 2018 Rolan Akhmedov. All rights reserved.
#

from __future__ import print_function

import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint

from dataset import Dataset, plot
from dataset import check_balance, make_balance

def train_and_save (dataset_path, model_path):
	# load train and test data
	ds = Dataset(dataset_path)
	ds.norm_amplitude()
	window_size = ds.max_pulse_wides()
	X, Y = ds.split(window_length=window_size, step=1)

	# prepare data for learning in keras
	for y in Y: y[0] = 0 if sum(y) > 1 else 1
	x_train = np.array(X[:9*len(X)//10]).astype('float32')
	y_train = np.array(Y[:9*len(Y)//10]).astype('float32')
	x_test  = np.array(X[9*len(X)//10:]).astype('float32')
	y_test  = np.array(Y[9*len(Y)//10:]).astype('float32')
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
	model.add(Dense(window_size//2, activation='relu', input_shape=(window_size,)))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

	history = model.fit(x_train, y_train,
						batch_size=256, epochs=10, verbose=1,
						validation_data=(x_test, y_test))

	model.save(model_path)

	# test the model on test data
	model.summary()
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	return history

# def	IoU (y_true, y_pred, gap = 10):
# 	iou = []
# 	while 
# 	return sum(iou) / len(iou)

def	IoU2 (y_true, y_pred, gap = 10):
	started = False; start = 0; iou = []
	for i in range(0,len(y_true)):
		if y_true[i][0] == 0:
			if not started:
				start = i
				started = True
			if started and not np.array_equal(y_true[start], y_true[i]):
					real = i - start
					pred = 0
					for j in range(start-gap,i+gap):
						if np.array_equal(y_true[start], y_pred[j]):
							pred = pred + 1
					started = False
					iou.append(min(pred, real)/max(pred,real))
	return sum(iou) / len(iou)

### MAIN ###

train_and_save('train.json','121_class.h5')

# load train and test data
ds = Dataset('test.json')
ds.norm_amplitude()
window_size = ds.max_pulse_wides()
_ , Y = ds.split(window_length=window_size, step=1)
for y in Y: y[0] = 0 if sum(y) > 1 else 1
Y = np.array(Y).astype('int32')
print(IoU2(Y,Y))


