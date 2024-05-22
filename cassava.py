#%%
import tensorflow as tf
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from history import plot_history, save_history
import tensorflow_datasets as tfds

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#%%
import tensorflow_datasets as tfds
# Load the cassava dataset

BATCH_SIZE = 32

training_ds, info = tfds.load('cassava', with_info=True, as_supervised=True, split='train')
test_ds = tfds.load('cassava', as_supervised=True, split='test')
validation_ds = tfds.load('cassava', as_supervised=True, split='validation')

# Define data augmentation operations
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Apply data augmentation
def augment_training(image, label):
    image = data_augmentation(image)
    image = tf.image.resize(image, (500, 500))
    return image, label

rescale_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
])

def augment_rescale(image, label):
    image = rescale_augmentation(image)
    image = tf.image.resize(image, (500, 500))
    return image, label

training_augmented = training_ds.map(augment_training)
test_augmented = test_ds.map(augment_rescale)
validation_augmented = validation_ds.map(augment_rescale)


training_augmented = training_augmented.batch(BATCH_SIZE)
test_augmented = test_augmented.batch(BATCH_SIZE)
validation_augmented = validation_augmented.batch(BATCH_SIZE)


#%%

#%%

#%%
def compile_model(model: tf.keras.Sequential):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model
#%%
def create_model():
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((500, 500, 3)),
        tf.keras.layers.experimental.preprocessing.Resizing(125, 125),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return compile_model(new_model)
#%%
def save_model(model, name, history, test_data):
    test_loss, test_acc = model.evaluate(test_data)

    # Save model information
    save_name = f'models/ibeans/{name}-{len(history.epoch):02d}-{test_acc:0.4f}'
    model.save(f'{save_name}.h5')

    # Save history information
    save_history(history, save_name)
#%%
model = create_model()

#%%
history = model.fit(training_augmented, epochs=5)
#%%
