FROM nvidia/cuda:8.0-cudnn7-devel

WORKDIR /app

ADD . /app

RUN apt-get update
RUN apt-get -qq install -y python2.7 libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev unzip zip cmake
RUN apt-get -qq install --no-install-recommends libboost1.58-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev wget python-pip git-all libzip-dev 

# RUN PYTHON INSTALL
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy scipy sklearn scikit-image


# Get code
RUN git clone --recursive -b docker_server https://github.com/yjxiong/anet2016-cuhk

WORKDIR /app/anet2016-cuhk

RUN git status

RUN nvcc --version

RUN bash -e build_all.sh

RUN bash models/get_reference_models.sh

CMD ["python", "demo_server.py"]
