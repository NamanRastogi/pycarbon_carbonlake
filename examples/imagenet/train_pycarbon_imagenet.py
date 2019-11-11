#  Copyright (c) 2018-2019 Huawei Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import division, print_function

import argparse
import os
import time

import jnius_config
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, decode_predictions
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import Adam

from examples import DEFAULT_CARBONSDK_PATH
from examples.imagenet.schema import ImagenetSchema
from pycarbon import make_carbon_reader


tf.logging.set_verbosity(tf.logging.ERROR)


def _args_parser():
  """Utility method for command-line argument parsing"""

  default_data_url = 'file:///tmp/imagenet/data/train'
  default_train_url = '/tmp/imagenet/train'
  default_vocab_url = '/tmp/imagenet/vocabulary.txt'

  parser = argparse.ArgumentParser(description='Pycarbon Tensorflow External ImageNet Example')
  parser.add_argument('--data-url',
                      type=str,
                      default=default_data_url,
                      help='hdfs:// or file:/// URL to the ImageNet PyCarbon dataset (default: %s)'.format(default_data_url))
  parser.add_argument('--train-url',
                      type=str,
                      default=default_train_url,
                      help='URL to directory to dump trained model (default: {})'.format(default_train_url))
  parser.add_argument('--num-epochs',
                      type=int,
                      default=10,
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--batch-size',
                      type=int,
                      default=4,
                      help='input batch size for training (default: 4)')
  parser.add_argument('--carbon-sdk-path',
                      type=str,
                      default=DEFAULT_CARBONSDK_PATH,
                      help='carbon sdk path')
  parser.add_argument('--vocab-url',
                      type=str,
                      default=default_vocab_url,
                      help='URL to directory to dump trained model (default: {})'.format(default_vocab_url))

  return parser


def decode(row):
  image = getattr(row, 'image')
  text = getattr(row, 'text')
  return text, image


def load_model(output_dimension):
  """Load Pre-Trained model, add dense layer and return model"""

  # Load Pre-Trained model
  base_model = ResNet50(include_top=False, weights='imagenet')
  for layer in base_model.layers:
    layer.trainable = False

  # Add new layers
  last_layer = GlobalAveragePooling2D()(base_model.output)
  last_layer = Dense(output_dimension, activation='softmax')(last_layer)
  finetuned_model = Model(inputs=base_model.input, outputs=last_layer)

  return finetuned_model


def encode_one_hot(label, vocabulary):
  return np.array([1.0 if label == word else 0.0 for word in vocabulary], dtype=np.float32)


def save_model(model, save_path):
  """Save model-weights on disk."""

  model_path = os.path.join(save_path, 'resnet50_finetuned_model.json')
  model_json_string = model.to_json()
  with open(model_path, mode='w') as model_write_file:
    model_write_file.write(model_json_string)

  weights_path  = os.path.join(save_path, 'resnet50_finetuned_weights.h5')
  model.save_weights(weights_path, overwrite=True, save_format='hdf5')


def train(model, data_url, num_epochs, batch_size, projection, vocabulary):
  """Training method."""

  schema_fields = [getattr(ImagenetSchema, field) for field in projection]

  # Train model
  with make_carbon_reader(data_url, schema_fields=schema_fields, num_epochs=num_epochs) as reader:
    # TODO: Read data in batch, and not each row individually
    for row in reader:
      image = np.expand_dims(row.image, axis=0)
      label = np.expand_dims(encode_one_hot(row.text.decode('utf-8'), vocabulary), axis=0)
      model.fit(image, label, steps_per_epoch=1)


def load_vocabulary(vocabulary_url):
  """Load vocabulary (can be done from a file)."""

  with open(vocabulary_url, mode='r') as vocab_read_file:
    return vocab_read_file.readline().strip().split()


def main(args):
  start_time = time.time()
  print('='*25, 'Started', '='*25)

  # Load vocabulary
  vocabulary = load_vocabulary(args.vocab_url)

  # Load ResNet50 model with pre-trained weights
  model = load_model(output_dimension=len(vocabulary))
  model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

  # Train model
  train(model=model,
        data_url=args.data_url,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        projection=['image', 'text'],
        vocabulary=vocabulary)

  # Save trained model
  save_model(model, args.train_url)

  print('='*25, 'Finished', '='*25)
  print('Time taken: {}'.format(str(time.time() - start_time)))


if __name__ == '__main__':
  parsed_args = _args_parser().parse_args()
  jnius_config.set_classpath(parsed_args.carbon_sdk_path)

  main(parsed_args)
