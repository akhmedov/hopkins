import os
import keras
import numpy as np
from sequental_lira_dataset import Dataset
import matplotlib.pyplot as plot
import matplotlib.ticker as ticker

# keras.layers.TimeDistributed
# keras.layers.SpatialDropout1D


def model_arch(input_dim, output_dim, batch_size, look_back):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(output_dim=output_dim, batch_input_shape=[batch_size, look_back, input_dim]))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.Dense(output_dim, init='uniform', activation='softmax'))
    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()
    return model


def train_model(num_epoch=20, batch_size=50, save_path='lstm_sequence_to_label.h5'):
    X, Y = Dataset.sequence_to_label('lira_sequental_dataset.json')
    X = np.expand_dims(X, axis=2)
    x_train = X[:9 * X.shape[0] // 10]
    y_train = Y[:9 * X.shape[0] // 10]
    x_valid = X[9 * X.shape[0] // 10:]
    y_valid = Y[9 * X.shape[0] // 10:]
    model = model_arch(input_dim=X.shape[2], output_dim=Y.shape[1], batch_size=batch_size, look_back=X.shape[1])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=True, mode='auto')
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                        callbacks=[early_stopping], nb_epoch=num_epoch,
                        batch_size=batch_size, verbose=True)
    model.save(save_path)
    return history


def plot_train_hist(history):
    sheet = plot.figure().gca()
    sheet.plot(history.history['loss'], label='Training')
    sheet.plot(history.history['val_loss'], label='Validation')
    sheet.set_title('model loss')
    sheet.set_ylabel('loss')
    sheet.set_xlabel('epoch')
    sheet.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    sheet.legend()
    plot.show()


def main(model_file='lstm_sequence_to_label.h5'):

    if not os.path.exists(model_file):
        history = train_model(save_path=model_file)
        plot_train_hist(history)

    X, Y = Dataset.sequence_to_sequence('lira_sequental_dataset.json')
    X = np.expand_dims(X, axis=2)
    model = keras.models.load_model(model_file)
    # score = model.evaluate(X, Y, verbose=True)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    y_predict = model.predict_classes(X, verbose=True)
    error = np.mean(y_predict != Y)
    print(error)


if __name__ == '__main__':
    main()
