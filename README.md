# CUHK & ETH & SIAT Solution to AcitivityNet Challenge 2016 

This repository holds the materials necessary to reproduce the results for our solution to ActivityNet Challenge 2016. 
We works on the untrimmed video classification task. 
Although initially designed for the challenge, the repository also means to provide an accessible framework for general video classification tasks.

*We are currently organizing the codebase. Please stay tuned.*

##Functionalities & Release Status

- [x] Basic utilities
- [x] Action recognition with single video
    * [ ] Web demo for action recognition
- [ ] ActivityNet validation set evaluation
- [ ] Training action recognition system

##Dependencies
The codebase is written in Python. It is recommended to use [Anaconda][anaconda] distribution package with it.
Besides, we also use Caffe and OpenCV. 
Particularly, the OpenCV should be compiled with VideoIO support. GPU support will be good if possible.

##Requirements
NVIDIA GPU with CUDA support, at least 4GB display memory.

##Get the code
Use Git
```
git clone --recursive https://github.com/yjxiong/anet2016-cuhk
```

If you happen to forget to add `--recursive` to the command. You can still go to the project directory and issue
```
git submodule update --init
```

##Single Video Classification
- Build all modules
In the root directory of the project, run the following command
```
bash build_all.sh
```
- Get reference models
```
bash models/get_reference_models.sh
```
- Run the classification
There is a video clip in the `data/plastering.avi` for your example.
To do single video classification with RGB model one can run
```
bash examples/classify_video.py data/plastering.avi
```
It should print the top 3 prediction in the output.
To use the two-stream model, one can add `--use_flow` flag to the command. The framework will then extract optical flow on the fly.
```
bash examples/classify_video.py --use_flow data/plastering.avi
```
You can use your own video files by specifying the filename. More functions to be added.
The two-stream model here consists of one reset-200 model for RGB input and one BN-Inception model for optical flow input.
The model spec and parameter files can be found in `models/`.

##LICENSE
Released under [BSD 2-Clause license][license].


[anaconda]:https://www.continuum.io/downloads
[license]:https://github.com/yjxiong/anet16-cuhk/blob/master/LICENSE