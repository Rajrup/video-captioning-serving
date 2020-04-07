import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import sys
import os
import numpy as np
from modules_video_cap.utils import *

sys.path.append(os.path.abspath('./'))

if (len(sys.argv) < 3):
  print("Less than 3 arguments")
  print("Argument 1: original/serving")
  print("Argument 2: vgg16/alexnet")
  exit()

if sys.argv[1] == 'original' or sys.argv[1] == 'serving': 
  mode = sys.argv[1]
else:
  print("Argument 1 invalid, pass original/serving")
  exit()

if sys.argv[2] == 'vgg16' or sys.argv[2] == 'alexnet':  
  chain = sys.argv[2]
else:
  print("Argument 2 invalid, pass vgg16/alexnet")
  exit()

if (mode == "original"):
  # original version
  from modules_video_cap.data_reader import DataReader
  from modules_video_cap.video_cap_vgg16 import VGG16
  from modules_video_cap.video_cap_alexnet import AlexNet
  from modules_video_cap.video_cap_s2vt import S2VT

elif (mode == "serving"):
  # serving version 
  from modules_video_cap.data_reader import DataReader
  from modules_video_cap.video_cap_vgg16_serving import VGG16
  from modules_video_cap.video_cap_alexnet_serving import AlexNet
  from modules_video_cap.video_cap_s2vt_serving import S2VT

# ============ Video Input Module ============
video_path = os.path.abspath("./modules_video_cap/Data/YoutubeClips/vid264.mp4")
reader = DataReader()
reader.Setup(video_path)

# ============ VGG16 Embedding Module ===========
if chain == 'vgg16':
  vgg16 = VGG16()
  vgg16.Setup()

# ============ AlexNet Embedding Module ===========
else:
  alexnet = AlexNet()
  alexnet.Setup()
  

# ============ S2VT Caption Module ===========
s2vt = S2VT()
s2vt.Setup()

while(True):

  # Read input
  frame_data = reader.PostProcess()
  if not frame_data:  # end of video 
    break

  if chain == 'vgg16':
    vgg16.PreProcess(frame_data)
    vgg16.Apply()
    features_data = vgg16.PostProcess()
  else:
    alexnet.PreProcess(frame_data)
    alexnet.Apply()
    features_data = alexnet.PostProcess()

  s2vt.PreProcess(features_data)
  s2vt.Apply()
  s2vt.PostProcess()

# ============ Play Video Module ============
play_video = raw_input('Play Video? ')
if play_video.lower() == 'y':
  playVideo(video_path)

  

    
