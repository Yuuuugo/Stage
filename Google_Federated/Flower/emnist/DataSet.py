# Copyright 2019, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Library for loading and preprocessing EMNIST training and testing data."""

import collections
from colorsys import yiq_to_rgb
from typing import Tuple

import tensorflow as tf
import tensorflow_federated as tff
import numpy as np

MAX_CLIENT_DATASET_SIZE = 418


def _reshape_for_digit_recognition(element):
  return (tf.expand_dims(element['pixels'], axis=-1), element['label'])


def _reshape_for_autoencoder(element):
  x = 1 - tf.reshape(element['pixels'], (-1, 28 * 28))
  return (x, x)


def create_preprocess_fn(
    num_epochs: int,
    batch_size: int,
    shuffle_buffer_size: int = MAX_CLIENT_DATASET_SIZE,
    emnist_task: str = 'digit_recognition',
    num_parallel_calls: tf.Tensor = tf.data.experimental.AUTOTUNE
) -> tff.Computation:
  """Creates a preprocessing function for EMNIST client datasets.

  The preprocessing shuffles, repeats, batches, and then reshapes, using
  the `shuffle`, `repeat`, `batch`, and `map` attributes of a
  `tf.data.Dataset`, in that order.

  Args:
    num_epochs: An integer representing the number of epochs to repeat the
      client datasets.
    batch_size: An integer representing the batch size on clients.
    shuffle_buffer_size: An integer representing the shuffle buffer size on
      clients. If set to a number <= 1, no shuffling occurs.
    emnist_task: A string indicating the EMNIST task being performed. Must be
      one of 'digit_recognition' or 'autoencoder'. If the former, then elements
      are mapped to tuples of the form (pixels, label), if the latter then
      elements are mapped to tuples of the form (pixels, pixels).
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.

  Returns:
    A `tff.Computation` performing the preprocessing discussed above.
  """
  if num_epochs < 1:
    raise ValueError('num_epochs must be a positive integer.')
  if shuffle_buffer_size <= 1:
    shuffle_buffer_size = 1

  if emnist_task == 'digit_recognition':
    mapping_fn = _reshape_for_digit_recognition
  elif emnist_task == 'autoencoder':
    mapping_fn = _reshape_for_autoencoder
  else:
    raise ValueError('emnist_task must be one of "digit_recognition" or '
                     '"autoencoder".')

  # Features are intentionally sorted lexicographically by key for consistency
  # across datasets.
  feature_dtypes = collections.OrderedDict(
      label=tff.TensorType(tf.int32),
      pixels=tff.TensorType(tf.float32, shape=(28, 28)))

  @tff.tf_computation(tff.SequenceType(feature_dtypes))
  def preprocess_fn(dataset):
    return dataset.shuffle(shuffle_buffer_size).repeat(num_epochs).batch(
        batch_size, drop_remainder=False).map(
            mapping_fn, num_parallel_calls=num_parallel_calls)

  return preprocess_fn


def get_centralized_datasets(
    train_batch_size: int = 20,
    test_batch_size: int = 500,
    train_shuffle_buffer_size: int = 10000,
    test_shuffle_buffer_size: int = 1,
    only_digits: bool = False,
    emnist_task: str = 'digit_recognition'
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Loads and preprocesses centralized EMNIST training and testing sets.

  Args:
    train_batch_size: The batch size for the training dataset.
    test_batch_size: The batch size for the test dataset.
    train_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the train dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the test dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    only_digits: A boolean representing whether to take the digits-only
      EMNIST-10 (with only 10 labels) or the full EMNIST-62 dataset with digits
      and characters (62 labels). If set to True, we use EMNIST-10, otherwise we
      use EMNIST-62.
    emnist_task: A string indicating the EMNIST task being performed. Must be
      one of 'digit_recognition' or 'autoencoder'. If the former, then elements
      are mapped to tuples of the form (pixels, label), if the latter then
      elements are mapped to tuples of the form (pixels, pixels).

  Returns:
    A tuple (train_dataset, test_dataset) of `tf.data.Dataset` instances
    representing the centralized training and test datasets.
  """
  if train_shuffle_buffer_size <= 1:
    train_shuffle_buffer_size = 1
  if test_shuffle_buffer_size <= 1:
    test_shuffle_buffer_size = 1

  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=only_digits)

  emnist_train = emnist_train.create_tf_dataset_from_all_clients()
  emnist_test = emnist_test.create_tf_dataset_from_all_clients()

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=1,#
      batch_size=train_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      emnist_task=emnist_task)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=test_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      emnist_task=emnist_task)

  emnist_train = train_preprocess_fn(emnist_train)
  emnist_test = test_preprocess_fn(emnist_test)

  return emnist_train, emnist_test


Train, Test = get_centralized_datasets()
import  tensorflow_datasets as tfds

def data_extration(Brute):
  Brute = tfds.as_numpy(Brute)
  L = []
  for example in Brute:
    L.append(example)
  X_  = []
  y_ = []
  for i in L:
    X_.append(i[0])
    y_.append(i[1])
  Better_X_ = []
  Better_y_ = []
  for j in range(len(X_)):
    for i in X_[j]:
      Better_X_.append(i)
  for j in range(len(y_)):
    for i in y_[j]:
      Better_y_.append(i)
  Better_X_ = np.array(Better_X_)
  Better_y_ = np.array(Better_y_)
  return Better_X_,Better_y_

X_train, y_train = data_extration(Train)
X_test, y_test = data_extration(Test)

