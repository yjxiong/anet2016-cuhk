# CUHK & ETH & SIAT Solution to ActivityNet Challenge 2016 

This repository holds the materials necessary to reproduce the results for our solution to ActivityNet Challenge 2016. 
We won the 1st place in the untrimmed video classification task. 

Although initially designed for the challenge, the repository also means to provide an accessible framework for general video classification tasks.

*We are currently organizing the codebase. Please stay tuned.*

* Jul 14 - The correct reference flow model is available for download. See [here](https://github.com/yjxiong/anet2016-cuhk/blob/master/models/get_reference_models.sh).
* Jul 11 - [Demo website][demo] is now online!
* Jul 10 - Web demo code released

##Functionalities & Release Status

- [x] Basic utilities
- [x] Action recognition with single video
    * [x] Web demo for action recognition
- [ ] ActivityNet validation set evaluation
- [x] Training action recognition system - We use the [TSN][tsn] framework to train our models.

##Dependencies
The codebase is written in Python. It is recommended to use [Anaconda][anaconda] distribution package with it.

Besides, we also use Caffe and OpenCV. 
Particularly, the OpenCV should be compiled with VideoIO support. GPU support will be good if possible.
If you use `build_all.sh`, it will locally install these dependencies for you.

##Requirements
NVIDIA GPU with CUDA support. At least 4GB display memory is needed to run the reference models.

##Get the code
Use Git
```
git clone --recursive https://github.com/yjxiong/anet2016-cuhk
```

If you happen to forget adding `--recursive` to the command. You can still go to the project directory and issue
```
git submodule update --init
```

##Single Video Classification
- Build all modules
In the root directory of the project, run the following command
```
bash build_all.sh
```
- Get the reference models
```
bash models/get_reference_models.sh
```
- Run the classification
There is a video clip in the `data/plastering.avi` for your example.
To do single video classification with RGB model one can run
```
python examples/classify_video.py data/plastering.avi
```
It should print the top 3 prediction in the output.
To use the two-stream model, one can add `--use_flow` flag to the command. The framework will then extract optical flow on the fly.
```
python examples/classify_video.py --use_flow data/plastering.avi
```
You can use your own video files by specifying the filename. 

One can also specify a youtube url here to do the classification, for example
```
python examples/classify_video.py https://www.youtube.com/watch?v=QkuC0lvMAX0
```

The two-stream model here consists of one reset-200 model for RGB input and one BN-Inception model for optical flow input. 
The model spec and parameter files can be found in `models/`.

###Web Demo

We also provide a light-weighted demo server. The server uses [Flask][flask].

```
python demo_server.py
```

It will be run on `127.0.0.1:5000`. It supports uploading local files and directly analyzing Youtube-style video urls.

For a quick start, we have set up a public demo server at

[Action Recognition Web Demo][demo]

The server runs on the Titan X GPU awarded for winning the challenge. Thanks to the organizers!

##Related Projects
* [Temporal Segment Networks (TSN)][tsn] models for the challenge are trained under the TSN framework.
* Our modified [Caffe][caffe] with fast parallel training and Video data IO
* [Dense Flow][df] toolkit for optical flow extraction
* [Very deep two-stream convnets][deep_2stream]
* [Trajectory-Pooled Deep-Convolutional Descriptors (TDD)][tdd]

##LICENSE
Released under [BSD 2-Clause license][license].


[anaconda]:https://www.continuum.io/downloads
[license]:https://github.com/yjxiong/anet2016-cuhk/blob/master/LICENSE
[flask]:http://flask.pocoo.org/
[demo]:http://action-demo.ie.cuhk.edu.hk/
[caffe]:https://github.com/yjxiong/caffe
[df]:https://github.com/yjxiong/dense_flow
[tdd]:https://github.com/wanglimin/TDD
[deep_2stream]:http://personal.ie.cuhk.edu.hk/~xy012/others/action_recog/
[tsn]:https://github.com/yjxiong/temporal-segment-networks

