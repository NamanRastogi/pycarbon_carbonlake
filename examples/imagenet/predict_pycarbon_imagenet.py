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

import jnius_config
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from examples import DEFAULT_CARBONSDK_PATH


tf.logging.set_verbosity(tf.logging.ERROR)


def _args_parser():
  """Utility method for command-line argument parsing"""

  default_data_url = '/tmp/imagenet/data/test'
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


def load_trained_model(trained_model_path):
  """Load Pre-Trained and Fine-tuned model"""

  # Load Model
  model_path = os.path.join(trained_model_path, 'resnet50_finetuned_model.json')
  with open(model_path, mode='r') as model_read_file:
    model_json_string = model_read_file.readline()
  model = model_from_json(model_json_string)

  # Load Weights
  weights_path = os.path.join(trained_model_path, 'resnet50_finetuned_weights.h5')
  model.load_weights(weights_path)

  return model


def load_reverse_vocabulary(vocabulary_url):
  """Load vocabulary from a file."""

  with open(vocabulary_url, mode='r') as vocab_file:
    vocab = vocab_file.readline().strip().split()

  reverse_vocab = {i: word for i, word in enumerate(vocab)}
  return reverse_vocab


def decode_predictions(predictions, reverse_vocabulary):
  """Return text label for the predictions"""

  labels = np.argmax(predictions, axis=1)
  return [reverse_vocabulary[label] for label in labels]


def main(args):
  # Load Model
  model = load_trained_model(args.train_url)

  # Load reverse dictionary
  reverse_vocabulary = load_reverse_vocabulary(args.vocab_url)

  # Load data from directory and predict labels
  image_generator_from_directory = ImageDataGenerator().flow_from_directory(args.data_url, target_size=(224, 224))
  predictions = model.predict_generator(image_generator_from_directory)
  labels = decode_predictions(predictions, reverse_vocabulary)
  for label in labels:
    print(label)


if __name__ == '__main__':
  parsed_args = _args_parser().parse_args()
  jnius_config.set_classpath(parsed_args.carbon_sdk_path)

  main(parsed_args)
