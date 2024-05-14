import tensorflow as tf
import pandas as pd
import os

iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}


# Task 1: Retrieve Iris Data from the Internet
def retrieve_iris_data():
    iris_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    cache_dir = '.'
    cache_subdir = "data"

    iris_file = tf.keras.utils.get_file('iris.data', iris_url,cache_dir=cache_dir, cache_subdir=cache_subdir)
    return iris_file


# Task 2: Prepare the Iris Data
def parse_iris_data(iris_path):
    iris_df = pd.read_csv(iris_path, names=iris_columns)
    iris_df.replace(label_map, inplace=True)
    return iris_df


# Task 3: Load the Iris Data into a TensorFlow Dataset
def convert_iris_to_dataset(iris_dataframe):
    features = iris_dataframe[iris_columns[:4]]
    labels = iris_dataframe[iris_columns[-1]]
    iris_dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    return iris_dataset


if __name__ == "__main__":
  iris_filepath = retrieve_iris_data()

  iris_data = parse_iris_data(iris_filepath)
  iris_ds = convert_iris_to_dataset(iris_data)
