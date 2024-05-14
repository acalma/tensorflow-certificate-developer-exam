import tensorflow as tf
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from history import plot_history, save_history, add_history, load_history

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def print_dict(json_dict, items=5):
    print({x: json_dict[x] for (i,x) in enumerate(json_dict) if i < items})


def retrieve_data():
    imdb_url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    cache_dir = 'data'
    cache_subdir = 'imdb'
    imdb_dir = f"{cache_dir}/{cache_subdir}/aclImdb"

    should_extract = not os.path.exists(imdb_dir) and not os.path.isdir(imdb_dir)

    tf.keras.utils.get_file('aclImdb_v1.tar.gz', imdb_url, extract=should_extract, cache_dir=cache_dir, cache_subdir=cache_subdir)

    imdb_train_dataset = tf.keras.preprocessing.text_dataset_from_directory(f"{imdb_dir}/train", label_mode="binary", batch_size=1,
                                                                            shuffle=True,seed=42)

    imdb_test_dataset = tf.keras.preprocessing.text_dataset_from_directory(f"{imdb_dir}/test", label_mode="binary", batch_size=1,
                                                                           shuffle=False)

    return imdb_train_dataset, imdb_test_dataset


def split_features_labels(dataset):
    features = []
    labels = []
    for feature, label in dataset:
        features.append(feature.numpy()[0].decode('utf-8').lower())
        labels.append(float(label.numpy()[0]))
    return features, labels


def wrangle_data(tokenizer, features, labels, sequence_length):
    tokens = tokenizer.texts_to_sequences(features)
    features_padded = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=sequence_length, padding='post',
                                                                    truncating='post')

    labels = np.array(labels)

    return features_padded, labels, tokens


def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def save_model(model: tf.keras.Sequential, model_name, history, test_data, test_labels):
    test_loss, test_accuracy = model.evaluate(test_data, test_labels)

    # Save the model
    save_name = f'models/reviews-{model_name}-{len(history.epoch)}-{test_accuracy:.4f}'
    model.save(f"{save_name}.tf", save_format="tf")

    # Save the history
    save_history(history, save_name)


def dnn_model(word_dimension, embedding_dimension, sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(word_dimension, embedding_dimension, input_length=sequence_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return compile_model(model)


def cnn_model(word_dimension, embedding_dimension, sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(word_dimension, embedding_dimension, input_length=sequence_length),
        tf.keras.layers.Conv1D(filters=128, kernel_size=8, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return compile_model(model)


def gru_model(word_dimension, embedding_dimension, sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(word_dimension, embedding_dimension, input_length=sequence_length),
        tf.keras.layers.GRU(embedding_dimension),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return compile_model(model)


def lstm_model(word_dimension, embedding_dimension, sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(word_dimension, embedding_dimension, input_length=sequence_length),
        tf.keras.layers.LSTM(embedding_dimension),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return compile_model(model)


def bidirectional_model(word_dimension, embedding_dimension, sequence_length):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(word_dimension, embedding_dimension, input_length=sequence_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dimension)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    return compile_model(model)


if __name__ == '__main__':
    train_dataset, test_dataset = retrieve_data()

    train_features , train_labels = split_features_labels(train_dataset)
    test_features, test_labels = split_features_labels(test_dataset)

    word_dimension = 3000 # about 95% fluency

    # oov == out of value token
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=word_dimension, oov_token='~~~')
    tokenizer.fit_on_texts(train_features)

    sequence_length = 100
    train_data, train_labels, train_tokens = wrangle_data(tokenizer, train_features, train_labels, sequence_length)
    test_data, test_labels, test_tokens = wrangle_data(tokenizer, test_features, test_labels, sequence_length)

    # DNN model
    model_name = "dnn"

    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'ckpts/reviews/{model_name}-'+'{epoch:02d}-{val_accuracy:.4f}')

    embedding_dimension = 32

    dnn = dnn_model(word_dimension, embedding_dimension, sequence_length)

    history_dnn = dnn.fit(train_data, train_labels, validation_split=0.1, epochs=10, callbacks=[earlystop, checkpoint],
                          batch_size=64)

    save_model(dnn, model_name, history_dnn, test_data, test_labels)
    save_history(history_dnn, model_name)

    # CNN model
    model_name = "cnn"

    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'ckpts/reviews/{model_name}-'+'{epoch:02d}-{val_accuracy:.4f}')
    embedding_dimension = 32

    cnn = cnn_model(word_dimension, embedding_dimension, sequence_length)

    history_cnn = cnn.fit(train_data, train_labels, validation_split=0.1, epochs=25, callbacks=[earlystop, checkpoint],
                            batch_size=64)

    save_model(cnn, model_name, history_dnn, test_data, test_labels)
    save_history(history_cnn, model_name)

    # GRU model
    model_name = "gru"

    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'ckpts/reviews/{model_name}-'+'{epoch:02d}-{val_accuracy:.4f}')
    embedding_dimension = 32

    gru = gru_model(word_dimension, embedding_dimension, sequence_length)

    history_gru = gru.fit(train_data, train_labels, validation_split=0.1, epochs=25, callbacks=[earlystop, checkpoint],
                            batch_size=64)

    save_model(gru, model_name, history_gru, test_data, test_labels)
    save_history(history_gru, model_name)

    # LSTM model
    model_name = "lstm"

    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'ckpts/reviews/{model_name}-'+'{epoch:02d}-{val_accuracy:.4f}')
    embedding_dimension = 32

    lstm = lstm_model(word_dimension, embedding_dimension, sequence_length)

    history_lstm = lstm.fit(train_data, train_labels, validation_split=0.1, epochs=25, callbacks=[earlystop, checkpoint],
                            batch_size=64)

    save_model(lstm, model_name, history_lstm, test_data, test_labels)
    save_history(history_lstm, model_name)


    # Bidirectional model
    model_name = "bidirectional"

    earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=3, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=f'ckpts/reviews/{model_name}-'+'{epoch:02d}-{val_accuracy:.4f}')
    embedding_dimension = 32

    bidirectional_dnn = bidirectional_model(word_dimension, embedding_dimension, sequence_length)

    history_bidirectional = bidirectional_dnn.fit(train_data, train_labels, validation_split=0.1, epochs=25, callbacks=[earlystop, checkpoint],
                            batch_size=64)

    save_model(bidirectional_dnn, model_name, history_bidirectional, test_data, test_labels)
    save_history(history_bidirectional, model_name)

    # --- Compare models ---

    # Load history
    history_dir = 'models'

    dnn_model_stub = 'reviews-dnn-6-0.8158'
    cnn_model_stub = 'reviews-cnn-6-0.8149'
    gru_model_stub = 'reviews-gru-6-0.8131'
    lstm_model_stub = 'reviews-lstm-6-0.8088'
    bidirectional_dnn = 'reviews-bidirectional-5-0.8122'

    history_dnn = load_history(f"{history_dir}/{dnn_model_stub}", model_format='.tf')
    dnn = history_dnn.model

    history_cnn = load_history(f"{history_dir}/{cnn_model_stub}", model_format='.tf')
    cnn = history_cnn.model

    history_gru = load_history(f"{history_dir}/{gru_model_stub}", model_format='.tf')
    gru = history_gru.model

    history_lstm = load_history(f"{history_dir}/{lstm_model_stub}", model_format='.tf')
    lstm = history_lstm.model

    history_bidirectional = load_history(f"{history_dir}/{bidirectional_dnn}", model_format='.tf')
    bidirectional = history_bidirectional.model
