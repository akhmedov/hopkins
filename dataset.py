#
#  time_series.py
#  Hopkins
#
#  Created by Rolan Akhmedov on 04.07.18.
#  Copyright © 2018 Rolan Akhmedov. All rights reserved.
#

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import json as js
import sys

class Dataset:

	# internal variables
	base = None
	series = None
	sparks_num = None

	# read dataset in json format
	def __init__ (self, path):
		file = open(path)
		data = js.load(file)
		self.sparks_num = data['sparks']
		self.series = data['series']
		self.base = int(data['radix'])

	# normalize the amplitude of the radiation for better learning
	def norm_amplitude (self):
		max_val = self.amplitude(0)
		for idx in self.series:
			val = idx['field']
			max_val = val if val > max_val else max_val
		for idx in self.series:
			idx['field'] = idx['field'] / max_val

	# splite time line for sequental overlayed windows (NO OBSERVER LOCATION)
	def split (self, window_length=200, trash_hold=0.707, step = 1):
		X = []; Y = []
		window_begin = 0
		while window_begin+window_length < self.length():
			window = []
			label = [0] * self.radix()
			for idx in range(window_begin,window_begin+window_length):
				window.append(self.amplitude(idx))
				for u in self.unit(idx):
					label[u] += 1
			for u in range(0,self.radix()):
				label[u] = 1 if label[u] > trash_hold * window_length else 0
			X.append(window)
			Y.append(label)
			window_begin += step
		return X, Y

	# minimum pulse wides as number of samples in timeline
	def min_pulse_wides (self):
		min_wides = self.max_pulse_wides()
		wides = 0
		for u in range(1,self.radix()):
			for idx in range(0,self.length()):
				if u in self.unit(idx):
					wides += 1
				else:
					if wides < min_wides and wides != 0:
						min_wides = wides
					wides = 0
		return min_wides

	# maximum pulse wides as number of samples in timeline
	def max_pulse_wides (self):
		max_wides = 0
		wides = 0
		for u in range(1,self.radix()):
			for idx in range(0,self.length()):
				if u in self.unit(idx):
					wides += 1
				else:
					if wides > max_wides:
						max_wides = wides
					wides = 0
		return max_wides

	# get time sequens
	def timeline (self, seconds=True):
		res = []
		for item in self.series:
			if seconds: res.append(item['vt'] / 299792458)
			else: res.append(item['vt'])
		return res

	def code (self):
		signal = []
		is_signal = False
		for idx in range(0,self.length()):
			unit = self.unit(idx)[-1]
			if not is_signal and unit: 
				is_signal = True
			if not unit and is_signal: 
				is_signal = False
				prev_unit = self.unit(idx-1)[-1]
				signal.append(prev_unit)
		return signal

	# get number of unique charaters including "zero" in the dataset
	def radix (self):
		return self.base

	# get number of charates that presents in the dataset
	def sparks (self):
		return self.sparks_num

	# get number of points in time series
	def length (self):
		return len(self.series)

	# get field amplitude
	def amplitude (self, series_idx):
		return self.series[series_idx]['field']

	# get relevant time
	def time (self, series_idx, seconds=True):
		if seconds: return self.series[series_idx]['vt'] / 299792458
		else: return self.series[series_idx]['vt']

	# get charecter at index
	def unit (self, series_idx):
		return self.series[series_idx]['unit']

	# get rho cordinate of signal source at index
	def rho (self, series_idx):
		return self.series[series_idx]['rho']

	# get phi cordinate of signal source at index
	def phi (self, series_idx):
		return self.series[series_idx]['phi']

	# get z cordinate of signal source at index
	def z   (self, series_idx):
		return self.series[series_idx]['z']


def test_sparks_number (sequental_ds):
	sparks = sequental_ds.sparks()
	counted = 0
	signal = False
	for idx in range(0,sequental_ds.length()):
		unit = sequental_ds.unit(idx)[-1]
		if not signal and unit: 
			signal = True
		if not unit and signal: 
			signal = False
			counted += 1
	return True if (sparks == counted) else False


def color (char_id):
	return {
		0 : 'white',
		1 : 'blue',
		2 : 'green',
		3 : 'black',
		4 : 'yellow'
	}[char_id]


def plot (X, Y, vt_max = None):
	return

def plot (dataset, vt_max = None):
	plt.figure()
	if vt_max: plt.xlim(0, vt_max)
	series = []
	time = dataset.timeline(False)
	from_idx = 0
	for vt in range(0,dataset.length()):
		field = dataset.amplitude(vt)
		series.append(field)
		if dataset.unit(vt)[-1] != 0:
			if from_idx == 0:
				from_idx = vt
		if (dataset.unit(vt)[-1] == 0):
			if from_idx != 0:
				col = color(dataset.unit(vt-1)[-1])
				plt.axvspan(time[from_idx], time[vt], facecolor=col, alpha=0.2)
			from_idx = 0
	plt.plot(time,series,color='r')
	plt.show()

def check_balance (Y):
	radix = len(Y[0])
	res = [0] * radix
	for y in Y: res += y
	return res / len(Y)

def make_balance (X,Y):
	newX = np.array([X[0].tolist()]).astype('float32')
	newY = np.array([Y[0].tolist()]).astype('float32')
	for i in range(1,len(Y)):
		balance = check_balance(newY)
		min_idx = balance.argmin()
		max_idx = balance.argmax()
		balanced = abs(balance[max_idx] - balance[min_idx]) < 0.1
		if balanced or (Y[i][max_idx] < 0.1 and not balanced):
			newY = np.append(newY, [Y[i]], axis=0)
			newX = np.append(newX, [X[i]], axis=0)
	return newX, newY

### MAIN ###

if len(sys.argv) > 1:
	ds = Dataset(sys.argv[1])
	ds.norm_amplitude()
	print("Min wides: ", ds.min_pulse_wides())
	print("Max wides: ", ds.max_pulse_wides())
	print("Length: ", ds.length())
	print("Sparks: ", ds.sparks())
	print("Sequence: ", ds.code())
	plot(ds,20)
