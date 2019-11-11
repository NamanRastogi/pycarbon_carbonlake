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
import uuid
from glob import glob

import grpc
from PIL import Image
import tensorflow as tf

from examples.imagenet import resnet50_prediction_pb2_grpc, resnet50_prediction_pb2


tf.logging.set_verbosity(tf.logging.ERROR)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(name)s : %(message)s')


def _args_parser():
  """Utility method for command-line argument parsing"""

  default_server_ip = 'localhost'
  default_port = '50061'
  default_images_url = '/tmp/images/'

  parser = argparse.ArgumentParser(description='Pycarbon Tensorflow External ImageNet Example')
  parser.add_argument('--server-ip',
                      type=str,
                      default=default_server_ip,
                      help='Server IP (default: {})'.format(default_server_ip))
  parser.add_argument('--server-port',
                      type=str,
                      default=default_port,
                      help='Server Port (default: {})'.format(default_port))
  parser.add_argument('--images-url',
                      type=str,
                      default=default_images_url,
                      help='URL to directory to images (default: {})'.format(default_images_url))

  return parser


class Client:
  """Client"""

  def __init__(self):
    self.logger = logging.getLogger(self.__class__.__name__)
    self.logger.setLevel(logging.DEBUG)


  @staticmethod
  def read_image(image_path):
    """Read image from disk"""

    image_height, image_width = 224, 224
    image_data = Image.open(image_path, 'r').resize((image_height, image_width), Image.BICUBIC)
    return image_data.tobytes(encoder_name='raw')


  @staticmethod
  def generate_images(dir_path):
    """Generates images from the directory"""

    file_extensions = ['*.jpg', '*.JPEG', '*.png', '*.bmp', '*.gif']
    for extension in file_extensions:
      for file in glob(os.path.join(dir_path, '**', extension), recursive=True):
        yield file


  def query_server(self, ip, port, dir_path):
    """Sends request to server and receives response"""

    # Establish connection with server
    self.logger.info('Connecting {}:{}'.format(ip, port))
    with grpc.insecure_channel('{}:{}'.format(ip, port)) as channel:
      stub = resnet50_prediction_pb2_grpc.ImageLabelPredictionStub(channel)

      # Create request for server
      for image_path in self.generate_images(dir_path):
        req_id = str(uuid.uuid4())
        image = self.read_image(image_path)
        self.logger.info('Requesting image label! Request ID: {}'.format(req_id))
        request = resnet50_prediction_pb2.Image(request_id=req_id,
                                                image=image)

        # Send request to server and accept response
        try:
          response = stub.get_prediction(request)
          resp_id = response.response_id
          label = response.label
          self.logger.info('Response received! Image: {} Response ID: {} Label: {}'.format(resp_id, image_path, label))
        except Exception as ex:
          self.logger.error('Error! Image: {} Exception: {}'.format(image_path, str(ex)))


def main(args):
  server_ip = args.server_ip
  server_port = args.server_port
  images_dir = args.images_url

  client = Client()
  client.query_server(server_ip, server_port, images_dir)


if __name__ == '__main__':
  parsed_args = _args_parser().parse_args()
  main(parsed_args)
