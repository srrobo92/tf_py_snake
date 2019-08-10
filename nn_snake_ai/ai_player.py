from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import os

from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# File names to use
train_data_fn = './TrainData.csv'
model_save_name = 'full_game.h5'

model_inputs = 5

# pandas to load the csv
dataframe = pd.read_csv(train_data_fn)
print(dataframe.head())

# Split train data into training, validation, and test data sets
train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

# Print lengths just to see the divisions
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_data_and_label(dataframe, shuf=True):
  dataframe = dataframe.copy()
  if shuf:
    dataframe = shuffle(dataframe)
  labels = dataframe.pop('target')
  ds = dataframe.values
  ds_labels = labels.values
  return ds, ds_labels


# Split the data into the data sets and the labels
train_ds, train_labels = df_to_data_and_label(train)
val_ds, val_labels = df_to_data_and_label(val, shuf=False)
test_ds, test_labels = df_to_data_and_label(test, shuf=False)


# Set up model, use model inputs for number of inputs to NN
# Two layers, 128 connections betwen each internal layer, to the output layer
model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(model_inputs)),
    layers.Dense(25, input_shape=(model_inputs,), activation='relu'),
    layers.Dense(25, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Set up the optimizer, type of loss, and meterics to look at
model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'],
              run_eagerly=True)

# Run the model through its training using the training data,
# and validate with the training data
model.fit(train_ds,
          train_labels,
          validation_data=(val_ds, val_labels),
          epochs=10)

# Save the end result of the model
# TODO: Should probably same incrementally just incase a previous model in traing was better
model.save(model_save_name)

# Use the test data to see how we do with new not seen data and print results
results = model.evaluate(test_ds, test_labels)
print(results)


# Use predict and loop through all the test data to see which ones we got wrong
predictions = model.predict(test_ds[:])
print(predictions)
