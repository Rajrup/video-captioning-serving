import os
import sys
import cv2
import numpy as np
import skimage.transform
import tensorflow as tf

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util

IMAGE_SIZE = 227
N_DIMS = 80

class AlexNet:

  def Setup(self):

    self.input = None

    # -- hyper settings
    self.image_size = IMAGE_SIZE
    self.keep_prob = np.float32(1.0)

    #mean of imagenet dataset in BGR
    self.imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

    # output
    self.features_fc7 = []

  def PreProcess(self, input):
    self.input = input

    # Convert image to float32 and resize to (227x227)
    self.input_image = cv2.resize(self.input['img'].astype(np.float32), (self.image_size, self.image_size))
    
    # Subtract the ImageNet mean
    self.input_image -= self.imagenet_mean

    # Reshape as needed to feed into model
    self.input_image = self.input_image.reshape((1, self.image_size, self.image_size, 3))

  def Apply(self):

    ichannel = grpc.insecure_channel("localhost:8500")
    self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)

    self.internal_request = predict_pb2.PredictRequest()
    self.internal_request.model_spec.name = 'alexnet'
    self.internal_request.model_spec.signature_name = 'predict_images'

    self.internal_request.inputs['input'].CopyFrom(
    tf.contrib.util.make_tensor_proto(self.input_image, shape=self.input_image.shape))
    self.internal_request.inputs['input_keep_prob'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.keep_prob, dtype=np.float32))

    self.internal_result = self.istub.Predict(self.internal_request, 10.0) # 10 sec timeout

    self.feature_fc7 = tensor_util.MakeNdarray(self.internal_result.outputs['output'])

  def PostProcess(self):

    self.features_fc7.extend(self.feature_fc7)
    
    if len(self.features_fc7) >= N_DIMS:
      curr_feats = np.array(self.features_fc7)
      output = {'features': curr_feats, 'num_features': curr_feats.shape[0], 'meta':self.input['meta']}
      self.features_fc7 = []
      return output
    else:
      return {'features': None, 'num_features': 0, 'meta': None}
      
  def log(self, s):
    print('[AlexNet] %s' % s)
  