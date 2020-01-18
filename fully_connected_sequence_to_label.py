#
#  121_classs.py
#  Hopkins
#
#  Created by Rolan Akhmedov on 22.09.18.
#  Copyright © 2018 Rolan Akhmedov. All rights reserved.
#

import os
import keras
import numpy as np
from sequental_lira_dataset import Dataset
import matplotlib.pyplot as plot
import matplotlib.ticker as ticker

def train_model(num_epoch=25, batch_size=50, save_path='fully_connected_sequence_to_label.h5'):
    X, Y = Dataset.sequence_to_label('lira_sequental_dataset.json')
    # X = np.expand_dims(X, axis=2)
    x_train = X[:9 * X.shape[0] // 10]
    y_train = Y[:9 * X.shape[0] // 10]
    x_valid = X[9 * X.shape[0] // 10:]
    y_valid = Y[9 * X.shape[0] // 10:]
    model = model_arch(output_dim=Y.shape[1], look_back=X.shape[1])
    callback_es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=True, mode='auto')
    callback_cp = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=True,
                                                  save_best_only=True, save_weights_only=False,
                                                  mode='auto', period=1)
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                        epochs=num_epoch,  callbacks=[callback_cp],
                        batch_size=batch_size, verbose=True)
    # model.save(save_path)
    return history


def model_arch(output_dim, look_back):
    model = keras.Sequential()
    model.add(keras.layers.Dense(look_back // 2, activation='relu', input_shape=(look_back, )))
    model.add(keras.layers.Dense(output_dim, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def average_error(model, X, Y):
    predict = model.predict_classes(X, verbose=True)
    Y = np.array([np.argmax(y) for y in Y])
    error = np.mean(predict != Y)
    return 100 * error


def get_wrong_cases(model, X, Y, snr):
    predict = model.predict_classes(X, verbose=True)
    Y = np.array([np.argmax(y) for y in Y])
    mask = np.bitwise_not(np.isclose(predict, Y))
    X, Y, predict, snr = X[mask], Y[mask], predict[mask], snr[mask]
    return X, Y, predict, snr


def plot_train_hist(history):
    sheet = plot.figure().gca()
    sheet.plot(history.history['loss'], label='Training', color='black', linestyle='--')
    sheet.plot(history.history['val_loss'], label='Validation', color='black', linestyle='-')
    sheet.grid(color='lightgrey', linestyle='--', linewidth=1)
    sheet.set_title('model loss')
    sheet.set_ylabel('loss')
    sheet.set_xlabel('epoch')
    sheet.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    sheet.legend()
    plot.show()


def main(model_file='fully_connected_sequence_to_label.h5', dataset_file='lira_sequental_dataset.json'):

    if not os.path.exists(model_file):
        history = train_model(save_path=model_file)
        plot_train_hist(history)

    X, Y = Dataset.sequence_to_label(dataset_file)
    # X = np.expand_dims(X, axis=2)

    random_model = model_arch(Y.shape[1], X.shape[1])
    trained_model = keras.models.load_model(model_file)

    random_model_error = average_error(random_model, X, Y)
    trained_model_error = average_error(trained_model, X, Y)

    print("Average error of random model on test dataset: {} %".format(random_model_error))
    print("Average error of trained model on test dataset: {} %".format(trained_model_error))

    # score = model.evaluate(X, Y, verbose=True)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    snr = Dataset.get_snr(dataset_file)
    labels = Dataset.class_labels(dataset_file)
    X, Y, predict, SNR = get_wrong_cases(trained_model, X, Y, snr)
    print("Wrong items: ", predict.shape)
    for y_true, y_false, x, snr in zip(Y, predict, X, SNR):
        text = 'Prediction: ' + labels[y_false] + '; Expert: ' + labels[y_true] + '; SNR = ' + str(snr)
        plot.plot(x, color='black', linestyle='-')
        plot.title(text)
        plot.ylabel('Ex')
        plot.grid(color='lightgrey', linestyle='--', linewidth=1)
        plot.show()


if __name__ == '__main__':
    main()
