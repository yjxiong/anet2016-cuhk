#!/usr/bin/env bash
# TODO: add compilation steps

# update the submodules: Caffe and Dense Flow
git submodule update --remote

# install common dependencies: OpenCV
# adpated from OpenCV.sh
version="2.4"

echo "Building OpenCV" $version
mkdir 3rd-party/
cd 3rd-party/

echo "Downloading OpenCV" $version
#wget -O OpenCV-$version.zip https://github.com/Itseez/opencv/archive/$version.zip
git clone --recursive -b 2.4 https://github.com/opencv/opencv opencv-$version

#echo "Installing OpenCV" $version
#unzip OpenCV-$version.zip
cd opencv-$version
mkdir build
cd build
git apply ../../opencv_cuda9.patch
cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON  -D WITH_V4L=ON ..
make -j32
cp lib/cv2.so ../../../
echo "OpenCV" $version "built"

# build dense_flow
cd ../../../

echo "Building Dense Flow"
cd lib/dense_flow
mkdir build 
cd build
OpenCV_DIR=../../../3rd-party/opencv-$version/build/ cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
make -j
echo "Dense Flow built"

# build caffe
echo "Building Caffe"
cd ../../caffe-action
mkdir build 
cd build
OpenCV_DIR=../../../3rd-party/opencv-$version/build/ cmake .. -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
make -j32
echo "Caffe Built"
cd ../../../

# install python packages
pip install -r py_requirements.txt

# setup for web demo
mkdir tmp

# copy website files to the folder
wget -O 3rd-party/bootstrap-fileinput.zip https://github.com/kartik-v/bootstrap-fileinput/zipball/master
cd 3rd-party
unzip bootstrap-fileinput.zip
mv kartik-v-bootstrap-* Bootstrap-fileinput
cp -r Bootstrap-fileinput/js ../static/js
cp Bootstrap-fileinput/css/* ../static/css/
