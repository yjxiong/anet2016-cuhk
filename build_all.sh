#!/usr/bin/env bash
# TODO: add compilation steps

# update the submodules: Caffe and Dense Flow
git submodule update --remote

# install Caffe dependencies
sudo apt-get -qq install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev
sudo apt-get -qq install --no-install-recommends libboost1.55-all-dev
sudo apt-get -qq install libgflags-dev libgoogle-glog-dev liblmdb-dev

# install Dense_Flow dependencies
sudo apt-get -qq install libzip-dev

# install common dependencies: OpenCV
# adpated from OpenCV.sh
version="2.4.12"

echo "Building OpenCV" $version
mkdir 3rd-party/
cd 3rd-party/

echo "Installing Dependenices"
sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils

echo "Downloading OpenCV" $version
wget -O OpenCV-$version.zip https://github.com/Itseez/opencv/archive/$version.zip

echo "Installing OpenCV" $version
unzip OpenCV-$version.zip
cd opencv-$version
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON  -D WITH_V4L=ON  -D WITH_QT=ON -D WITH_OPENGL=ON ..
make -j32
cp lib/cv2.so ../../../
echo "OpenCV" $version "built"

# build dense_flow
cd ../../../

echo "Building Dense Flow"
cd lib/dense_flow
mkdir build 
cd build
OpenCV_DIR=../../../3rd-party/opencv-$version/build/ cmake ..
make -j
echo "Dense Flow built"

# build caffe
echo "Building Caffe"
cd ../../caffe-action
mkdir build 
cd build
OpenCV_DIR=../../../3rd-party/opencv-$version/build/ cmake ..
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