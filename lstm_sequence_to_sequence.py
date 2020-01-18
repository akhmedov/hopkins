import os
import keras
import numpy as np
from sequental_lira_dataset import Dataset
import matplotlib.pyplot as plot
import matplotlib.ticker as ticker


def model_arch(input_dim, output_dim, look_back):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=output_dim, return_sequences=True, stateful=False,
                                batch_input_shape=[None, look_back, input_dim]))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.TimeDistributed(
        keras.layers.Dense(output_dim, kernel_initializer='uniform', activation='softmax')))
    model.compile(loss='mse', optimizer='rmsprop')
    model.summary()
    return model


def train_model(save_path, dataset_path, num_epoch=150, batch_size=50):
    X, Y = Dataset.sequence_to_sequence(dataset_path)
    X = np.expand_dims(X, axis=2)
    Y = Y.transpose([0, 2, 1])
    x_train = X[:9 * X.shape[0] // 10]
    y_train = Y[:9 * X.shape[0] // 10]
    x_valid = X[9 * X.shape[0] // 10:]
    y_valid = Y[9 * X.shape[0] // 10:]
    model = model_arch(input_dim=X.shape[2], output_dim=Y.shape[1], look_back=X.shape[1])
    callback_es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=True, mode='auto')
    callback_cp = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=True,
                                                  save_best_only=True, save_weights_only=False,
                                                  mode='auto', period=1)
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                        epochs=num_epoch,  callbacks=[callback_cp],
                        batch_size=batch_size, verbose=True)
    return history


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


def main(model_file='lstm_sequence_to_sequence.h5', dataset_file='lira_sequental_dataset.json'):

    if not os.path.exists(model_file):
        history = train_model(save_path=model_file, dataset_path=dataset_file)
        plot_train_hist(history)

    X, Y = Dataset.sequence_to_sequence(dataset_file)
    X = np.expand_dims(X, axis=2)
    Y = Y.transpose([0, 2, 1])

    random_model = model_arch(X.shape[2], Y.shape[2], X.shape[1])
    trained_model = keras.models.load_model(model_file)

    prediction = trained_model.predict(np.array([X[0]]))
    print(prediction[0])


if __name__ == '__main__':
    main()
