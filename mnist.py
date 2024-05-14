import tensorflow as tf
import tensorflow_datasets as tfds
import os


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def wrangle_data (dataset: tf.data.Dataset, split):
    wrangled = dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    wrangled = wrangled.cache()
    if split == 'train':
        wrangled = wrangled.shuffle(60000)
    return wrangled.batch(64).prefetch(tf.data.AUTOTUNE)


def compile_model(model: tf.keras.Sequential):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model)
    return model


def create_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((28,28,1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return compile_model(new_model)


if __name__ == "__main__":
    # with Tensorflow Datasets
    training_ds , mnist_info = tfds.load('mnist', split='train', shuffle_files=True, as_supervised=True, with_info=True)
    test_ds = tfds.load('mnist', split='test', shuffle_files=False, as_supervised=True)

    assert isinstance(training_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)

    # tfds.show_examples(training_ds, mnist_info)

    train_data = wrangle_data(training_ds, 'train')
    test_data = wrangle_data(test_ds, 'test')

    # Train a model
    model = create_model()

    history = model.fit(train_data, epochs=5)


