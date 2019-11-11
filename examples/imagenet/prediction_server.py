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


import argparse
import logging
import os
import time
import uuid
from concurrent import futures

import grpc
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import model_from_json

from examples.imagenet import resnet50_prediction_pb2_grpc, resnet50_prediction_pb2

tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s : %(message)s')
_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def _args_parser():
  """Utility method for command-line argument parsing"""

  default_train_url = '/tmp/imagenet/train'
  default_vocab_url = '/tmp/imagenet/vocabulary.txt'
  default_port = '50061'
  default_num_workers = 2

  parser = argparse.ArgumentParser(description='Pycarbon Tensorflow External ImageNet Example')
  parser.add_argument('--train-url',
                      type=str,
                      default=default_train_url,
                      help='URL to directory to trained model (default: {})'.format(default_train_url))
  parser.add_argument('--vocab-url',
                      type=str,
                      default=default_vocab_url,
                      help='URL to directory to vocabulary file (default: {})'.format(default_vocab_url))
  parser.add_argument('--port',
                      type=str,
                      default=default_port,
                      help='Port to server listen to (default: {})'.format(default_port))
  parser.add_argument('--num-workers',
                      type=int,
                      default=default_num_workers,
                      help='number of workers (default: 10)')

  return parser


def load_trained_model(trained_model_path):
  """Load Pre-Trained and Fine-tuned model"""

  print('Model Loaded !')

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

  with open(vocabulary_url, mode='r') as vocab_read_file:
    vocab = vocab_read_file.readline().strip().split()

  reverse_vocab = {i: word for i, word in enumerate(vocab)}
  return reverse_vocab


def decode_predictions(predictions, reverse_vocabulary):
  """Return text label for the predictions"""

  assert predictions.shape[0] == 1, 'expected shape of predictions (1, ?), found '.format(predictions.shape)
  label = np.argmax(predictions[0])
  return reverse_vocabulary[label]


class ImageLabelPredictionServicerImpl(resnet50_prediction_pb2_grpc.ImageLabelPredictionServicer):
  """Service provider to ImageSimilarity"""

  def __init__(self, trained_model_url, vocab_url):
    self.logger = logging.getLogger(self.__class__.__name__)
    self.logger.setLevel(logging.DEBUG)
    self.trained_model_url = trained_model_url
    # self.model = load_trained_model(trained_model_url)
    # self.model._make_predict_function()
    self.reverse_vocabulary = load_reverse_vocabulary(vocab_url)


  def get_prediction(self, request, context):
    """Receives input from client and sends output"""

    # Accept request from client
    self.logger.info('Request received! Request ID: {}'.format(request.request_id))
    image_height, image_width = 224, 224
    image_pil = Image.frombytes(mode='RGB', size=(image_height, image_width), data=request.image, decoder_name='raw')
    image = np.expand_dims(np.asarray(image_pil), axis=0)

    # Generate label for image using "ResNet50 for Image Classification"
    # TODO: Load model once in __init__() and use in get_predictions()
    # TODO: instead of loading it again and again here
    model = load_trained_model(self.trained_model_url)
    prediction = model.predict(image)
    label = decode_predictions(prediction, self.reverse_vocabulary)

    # Send response back to client
    resp_id = str(uuid.uuid4())
    self.logger.info('Sending response! Request ID: {} Response ID: {} Similarity: {}'.format(request.request_id, resp_id, label))
    return resnet50_prediction_pb2.Label(
      response_id=resp_id,
      label=label
    )


class Server:
  """Server Manager"""

  def __init__(self, trained_model_url, vocab_url, num_workers):
    self.logger = logging.getLogger(self.__class__.__name__)
    self.logger.setLevel(logging.DEBUG)
    self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=num_workers))
    resnet50_prediction_pb2_grpc.add_ImageLabelPredictionServicer_to_server(
      ImageLabelPredictionServicerImpl(trained_model_url, vocab_url),
      self.server)


  def start(self, port):
    """Start server"""

    self.server.add_insecure_port('[::]:{}'.format(port))
    self.server.start()
    self.logger.info('Server started at port {}'.format(port))
    try:
      while True:
        time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
      self.server.stop(0)
      self.logger.info('Server stopped!')


def main(args):
  trained_model_url = args.train_url
  vocab_url = args.vocab_url
  num_workers = args.num_workers
  port = args.port

  server = Server(trained_model_url, vocab_url, num_workers)
  server.start(port=port)


if __name__ == '__main__':
  parsed_args = _args_parser().parse_args()
  main(parsed_args)
