import json

import numpy as np
import pandas as pd
import tensorflow as tf


import pandas as pd
import matplotlib.pyplot as plt


def plot_precipitation_data(data):
    """
    Plot the precipitation data.

    Args:
    data (DataFrame): DataFrame containing precipitation data.
    """
    # Convert data types
    # data["Date"] = pd.to_datetime(data["Date"])
    data["Precipitation"] = data["Precipitation"].astype(float)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(data["Date"], data["Precipitation"], marker='o', linestyle='-')
    plt.title("Monthly Precipitation")
    plt.xlabel("Date")
    plt.ylabel("Precipitation (Inches)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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


# Step 1: Retrieve and preprocess the data
# Assuming you have loaded the data from NOAA's Climate at a Glance and into a DataFrame named 'precipitation_data'

# Read the JSON file into a DataFrame
with open("data/precipitation.json", "r") as file:
    data = json.load(file)

# Extract the data part of the JSON
data = data["data"]

# Convert the data to a DataFrame
precipitation_data = pd.DataFrame.from_dict(data, orient="index")
precipitation_data.reset_index(inplace=True)
precipitation_data.columns = ["Date", "Precipitation"]
precipitation_data["Year"] = precipitation_data["Date"].str[:4].astype(int)
precipitation_data["Month"] = precipitation_data["Date"].str[4:].astype(int)

# Display the first few rows of the DataFrame
print(precipitation_data.head())

# Assuming 'precipitation_data' DataFrame is already loaded with the data
plot_precipitation_data(precipitation_data)


# Normalize the precipitation values using TensorFlow's normalize function
# Min-Max normalize the precipitation values
min_val = precipitation_data['Precipitation'].min()
max_val = precipitation_data['Precipitation'].max()
normalized_precipitation = (precipitation_data['Precipitation'] - min_val) / (max_val - min_val)

# Assign normalized values back to the DataFrame
precipitation_data['Precipitation (normalized)'] = normalized_precipitation


# Filter data for the years 1900 to 1999 (inclusive)
training_data = precipitation_data[(precipitation_data['Year'] >= 1900) & (precipitation_data['Year'] <= 1999)]


# Split the data into sequences of 6 months for training
sequence_length = 24
sequences = []
for i in range(len(training_data) - sequence_length + 1):
    sequences.append(training_data['Precipitation (normalized)'].values[i:i+sequence_length])

# Convert sequences to numpy array
sequences = np.array(sequences)

# Split into input (X) and target (y) variables
X_train = sequences[:, :-1]
y_train = sequences[:, -1]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Step 2: Design the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(30, kernel_size=6, padding='causal', activation='relu'),
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)

])

# Compile the model
model.compile(optimizer='adam', loss='mae', metrics=[tf.keras.metrics.RootMeanSquaredError(name="RMSE")])

# Step 4: Train the model
earlystop = tf.keras.callbacks.EarlyStopping('val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[earlystop])

# Step 5: Evaluate the model
# Assuming you have loaded the test data into a DataFrame named 'test_data'
# Filter data for the years 1900 to 1999 (inclusive)
test_data = precipitation_data[(precipitation_data['Year'] >= 2000) & (precipitation_data['Year'] <= 2020)]

# Prepare test sequences
test_sequences = []
for i in range(len(test_data) - sequence_length + 1):
    test_sequences.append(test_data['Precipitation (normalized)'].values[i:i+sequence_length])

test_sequences = np.array(test_sequences)

X_test = test_sequences[:, :-1]
y_test = test_sequences[:, -1]

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict
y_pred = model.predict(X_test)


# Convert predicted values and ground truth labels to tensors
y_true_tensor = tf.constant(test_data['Precipitation (normalized)'].values[sequence_length-1:], dtype=tf.float32)
y_pred_tensor = tf.constant(y_pred.flatten(), dtype=tf.float32)

# Calculate mean squared error
mse = tf.keras.losses.mean_squared_error(y_true_tensor, y_pred_tensor)

# Calculate RMSE
rmse = tf.sqrt(mse)

# Print RMSE
print("Root Mean Squared Error:", rmse.numpy())
