import os
import sys
import cv2
import numpy as np
import skimage.transform
import tensorflow as tf

# Path to actual model
ALEXNET_CKPT = os.path.abspath('./modules_video_cap/alexnet/model/alexnet_frozen.pb')

IMAGE_SIZE = 227
N_DIMS = 80

class AlexNet:
  def __init__(self):

    self.input = None

    # -- hyper settings
    self.image_size = IMAGE_SIZE
    
    #mean of imagenet dataset in BGR
    self.imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

    # output
    self.features_fc7 = []

  def Setup(self):
    
    self.log('Loading model...')
    
    self.alexnet_graph = tf.Graph()

    # loading network
    with self.alexnet_graph.as_default() as sg:
      with tf.gfile.GFile(ALEXNET_CKPT, 'rb') as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())
          g_in = tf.import_graph_def(graph_def)
          self.log('Model loading complete!')

    self.sess = tf.Session(graph = self.alexnet_graph)

    # net inputs and outputs
    self.keep_prob = self.alexnet_graph.get_tensor_by_name("import/Const:0") # keep_prob always as 1.0
    self.image = self.alexnet_graph.get_tensor_by_name('import/Placeholder:0')
    self.fc7 = self.alexnet_graph.get_tensor_by_name('import/fc7/fc7:0')


  def PreProcess(self, input):
    self.input = input

    # Convert image to float32 and resize to (227x227)
    self.input_image = cv2.resize(self.input['img'].astype(np.float32), (self.image_size, self.image_size))
    
    # Subtract the ImageNet mean
    self.input_image -= self.imagenet_mean

    # Reshape as needed to feed into model
    self.input_image = self.input_image.reshape((1, self.image_size, self.image_size, 3))

  def Apply(self):
    if not self.input:
      self.log('Input is empty')
      return 

    feature = self.sess.run([self.fc7], feed_dict={self.image: self.input_image})

    if len(feature) != 0:
      self.feature_fc7 = feature[0]
    else:
      self.log('Error while extracting FC7 embedding')
      return 

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
  