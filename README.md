# Inference for Video Captioning using Tensorflow and Tensorflow Serving
Based on https://github.com/vijayvee/video-captioning<br/>
Paper: [Sequence to Sequence -- Video to Text](http://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf)


## How to Use
- Clone this repo to your machine ```git clone --recurse-submodules https://github.com/Rajrup/video-captioning-serving.git```
- Download the checkpoints and tensorflow servables (see below)
- Double check the model paths
- To Run ```pipeline.py```

## Requirements
For now we support ```Python 2.7``` (Support for ```Python 3``` will be added soon)
See ```requirements.txt``` for python packages.
```
apt install libopencv-dev
pip install -r requirements.txt
```

## Checkpoint and Data Preparation
- [Google Drive](https://drive.google.com/open?id=1KKGOtrcrrlmmg55J1GbdHJtpgtY5os1x)
- All the checkpoints for tensorflow models files should be put into the "modules_video_cap/" folder.
  + VGG16: Path will look this - ```./modules_video_cap/vggnet/model/```
  + AlexNet: Path will look this - ```./modules_video_cap/alexnet/model/```
  + S2VT:  Path will look this - ```./modules_video_cap/s2vt/model/```

- Tensorflow serving models (```tf_servable/```) should be put into the current folder.
- Data path will look this - ```./modules_video_cap/Data/```
- Change the ```source``` in ```run_tf_server.sh``` to absolute path to ```tf_servable/``` folder.

## Running
- Tensorflow Pipeline:
  ```
  python pipeline.py original
  ``` 
- Tensorflow Serving Pipeline:
  + Run serving in docker
    ```
    chmod +x run_tf_server.sh
    ./run_tf_server.sh
    ```
  + Run client:
    ```
    python pipeline.py serving
    ``` 

## Components
One module's output will go to the next one
- Video Frame Embeddings - VGG16
- Sequence to Text - S2VT

## Performance
- VGG16: *TODO*
- AlexNet: *TODO*
- S2VT: *TODO*
