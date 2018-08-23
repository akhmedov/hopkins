#
#  lstm_binary_labelling.py
#  Hopkins
#
#  Created by Rolan Akhmedov on 05.07.18.
#  Copyright © 2018 Rolan Akhmedov. All rights reserved.
#

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM, SpatialDropout1D
from keras.layers import TimeDistributedDense

'''
 - sequence clasification
 - sequence labeling
 - natural language processing
'''

# max_features = 5000
# maxlen = 80
# (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=max_features)
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
# model.add(Embedding(max_features, 32))
# model.add(SpatialDropout1D(0.2))
# model.add(Dense(1, activation="sigmoid"))

# Fixed seed for reproducibility of results
np.random.seed(42)

# Create a network
model = Sequential()

# Layer for long-short memory: dropout=0.2 ???
model.add(LSTM(input_shape=(emb,), input_dim=emb, 
	output_dim=hidden, return_sequences=True))

# WHAT IS THIS???
model.add(TimeDistributedDense(output_dim=2))

# Classification layer
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Learn the model: epochs=7 vs nb_epoch=3 ???
model.fit(X_train, Y_train, batch_size=32, nb_epoch=3, 
		  validation_data=(X_test, Y_test), verbose=1, show_accuracy=True)

# Test the trained model
scores = model.evaluate(X_test, Y_test, batch_size=64)
print("Test accuracy (inversed loss): %.2f%%" % (scores[1] * 100))
print(scores)
