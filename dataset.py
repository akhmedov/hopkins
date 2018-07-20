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

class Dataset:

	# internal variables
	radix = 4
	series = 0
	sparks_num = 0

	# read dataset in json format
	def __init__ (self, path):
		file = open(path)
		data = js.load(file)
		self.sparks_num = data['sparks']
		self.series = data['series']
		self.radix = int(data['radix'])

	# get time sequens
	def timeline (self):
		res = []
		for item in self.series:
			res.append(item['vt'] / 0.299792458)
		return res

	def code (self):
		signal = []
		is_signal = False
		for idx in range(0,ds.length()):
			unit = ds.unit(idx)[-1]
			if not is_signal and unit: 
				is_signal = True
			if not unit and is_signal: 
				is_signal = False
				prev_unit = ds.unit(idx-1)[-1]
				signal.append(prev_unit)
		return signal

	# get number of unique charaters including "zero" in the dataset
	def radix (self):
		return self.radix

	# get number of charates that presents in the dataset
	def sparks (self):
		return self.sparks_num

	# get number of points in time series
	def length (self):
		return len(self.series)

	# get field amplutude
	def amplitude (self, series_idx):
		return self.series[series_idx]['field']

	# get relevant time
	def time (self, series_idx):
		return self.series[series_idx]['vt'] / 0.299792458

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


def plot (dataset, vt_max = None):
	
	plt.figure()
	if vt_max: plt.xlim(0, vt_max)
	series = []
	time = dataset.timeline()
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


### MAIN ###


ds = Dataset('dataset.json')
print("Length: ", ds.length())
print("Sparks: ", ds.sparks())
print("Sequence: ", ds.code())
plot(ds,50)

