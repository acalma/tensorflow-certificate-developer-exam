import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.src.preprocessing.image import ImageDataGenerator
from keras.src.engine.sequential import Sequential
from keras.src.callbacks import History
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from history import plot_history, save_history, add_history, load_history


def retrieve_data():
    traffic_data_path = 'data/Metro_Interstate_Traffic_Volume.csv'
    return traffic_data_path


def plot_sequence(time, sequences, start=0, end=None):
    y_max = 1.0

    if len(np.shape(sequences)) == 1:
        sequences = [sequences]

    time = time[start:end]
    plt.figure(figsize=(28, 28))

    for seq in sequences:
        y_max = max(np.max(seq), y_max)
        seq = seq[start:end]
        plt.plot(time, seq)

    plt.ylim(0, y_max)
    plt.xlim(np.min(time), np.max(time))
    plt.show()


def show_predictions(trained_model, predict_sequence, true_values, predict_time, begin=0, end=None):
    predictions = trained_model.predict(predict_sequence)
    predictions = predictions[:, -1].reshape(len(predictions))
    plot_sequence(predict_time, (true_values, predictions), begin, end)
    return predictions


def normalize(values: np.ndarray):
    max_value = np.max(values)
    min_value = np.min(values)

    return (values - min_value) / (max_value - min_value)


def split_data(values: np.ndarray, time: np.ndarray, split_size=0.8):
    d_split = int(np.ceil(len(values) * split_size))

    big_values = values[:d_split]
    big_time = time[:d_split]

    small_values = values[d_split:]
    small_time = time[d_split:]

    print(f'Splitting into {len(big_values)} main samples and {len(small_values)} extra samples.')

    return big_values, big_time, small_values, small_time


def wrangle_data(sequence, data_split, examples, batch_size):
    # add one datapoint -- the one to predict
    examples = examples + 1

    # add a rank for the data points
    seq_expand = tf.expand_dims(sequence, -1)

    # create dataset
    dataset = tf.data.Dataset.from_tensor_slices(seq_expand)

    # window changes the data from a single sequence to a list of example-sized sequences
    dataset = dataset.window(examples, shift=1, drop_remainder=True)

    # transform the list of windowed sequences back to a list of sequences via batching
    dataset = dataset.flat_map(lambda b: b.batch(examples))

    # split data into features and labels
    dataset = dataset.map(lambda x: (x[:-1], x[-1]))

    if data_split == 'train':
        # shuffle the training data to improve generalization
        dataset = dataset.shuffle(10000)
    else:
        # cache non-training datasets for improved performance
        dataset = dataset.cache()

    # create batches from the data to feed to the model
    dataset = dataset.batch(batch_size)

    # prefetch for better performance. No caching because we want the data to be reshuffled each epoch.
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def compile_model(new_model: tf.keras.models.Sequential, loss='mse'):
    new_model.compile(optimizer='adam',
                      loss=loss,
                      metrics=['mae'])
    print(new_model.summary())
    return new_model


def dnn_model():
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer((None, 1)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return compile_model(new_model)


def bigger_dnn_model():
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer((None, 1)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return compile_model(new_model)


def cnn_model():
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer((None, 1)),
        tf.keras.layers.Conv1D(30, kernel_size=6, padding='causal', activation='relu'),
        tf.keras.layers.Conv1D(30, kernel_size=6, padding='causal', activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return compile_model(new_model)


def rnn_model():
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer((None, 1)),
        tf.keras.layers.Conv1D(30, kernel_size=6, padding='causal', activation='relu'),
        tf.keras.layers.LSTM(60),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return compile_model(new_model)


def stacked_rnn_model():
    new_model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer((None, 1)),
        tf.keras.layers.Conv1D(30, kernel_size=6, padding='causal', activation='relu'),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return compile_model(new_model)


def save_model(model: Sequential, name, history: History, test_data):
    test_loss, test_acc = model.evaluate(test_data)

    # Save model information
    save_name = f'models/{name}-{len(history.epoch):02d}-loss-{model.loss}-{history.history["loss"][-1]}-test-acc-{test_acc:0.4f}'
    model.save(f'{save_name}.h5')

    # Save history information
    save_history(history, save_name)

def load_model(checkpoint_dir: str):
    model = tf.keras.models.load_model(checkpoint_dir)
    return model


if __name__ == '__main__':
    traffic_file = retrieve_data()

    # load data
    traffic_df = pd.read_csv(traffic_file)
    traffic_volume = np.array(traffic_df['traffic_volume'])
    time_steps = np.array(list(range(len(traffic_volume))))

    # vars for visualization
    day = 24
    week = 7 * day
    month = 4 * week

    # preprocessing
    normalized_traffic = normalize(traffic_volume)

    # split
    train_volume_tmp, train_time_tmp, test_volume, test_time = split_data(normalized_traffic, time_steps,
                                                                          split_size=0.8)

    train_volume, train_time, validation_volume, validation_time = split_data(train_volume_tmp, train_time_tmp,
                                                                              split_size=0.9)

    # sequences for the model
    examples = 24
    batch_size = 30

    train_data = wrangle_data(train_volume, 'train', examples, batch_size)
    validation_data = wrangle_data(validation_volume, 'valid', examples, batch_size)
    test_data = wrangle_data(test_volume, 'test', examples, batch_size)

    # small model
    # model_name = 'dnn_traffic_regression'
    # earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'ckpts/traffic/{model_name}/' + '{epoch:02d}-{val_loss:.4f}'
    # )
    # dnn = dnn_model()


    # bigger model
    # model_name = 'dnn_bigger_traffic_regression'
    # earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'ckpts/traffic/{model_name}/' + '{epoch:02d}-{val_loss:.4f}'
    # )
    #
    # dnn = bigger_dnn_model()

    # cnn model
    # model_name = 'cnn_traffic_regression'
    # earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=f'ckpts/traffic/{model_name}/' + '{epoch:02d}-{val_loss:.4f}'
    # )
    #
    # dnn = cnn_model()

    # train
    # history = dnn.fit(
    #     train_data,
    #     validation_data=validation_data,
    #     callbacks=[earlystop, checkpoint],
    #     epochs=10
    # )

    #save_model(dnn, model_name, history, test_data)
