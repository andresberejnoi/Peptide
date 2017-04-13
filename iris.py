from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "curated_train_set.csv"
IRIS_TEST = "testdata.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TRAINING,
                                                       target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=IRIS_TEST,
                                                   target_dtype=np.int)
# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=5)]

#Build 3 layer DNN with 10,20,10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=2,
                                            model_dir="/tmp/iris_model")
# Define the test inputs
def get_train_inputs():
  x = tf.constant(training_set.data)
  y = tf.constant(training_set.target)

  return x, y

# Fit model.
classifier.fit(input_fn=get_train_inputs, steps=2000)

classifier.fit(x=training_set.data, y=training_set.target, steps=1000)
classifier.fit(x=training_set.data, y=training_set.target, steps=1000)

# Define the test inputs
def get_test_inputs():
  x = tf.constant(test_set.data)
  y = tf.constant(test_set.target)

  return x, y

# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                     steps=1)["accuracy"]

print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
