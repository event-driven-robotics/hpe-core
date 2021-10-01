
# base image
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
# FROM ubuntu

###########
# OPENPOSE #
###########

# ARG OPENPOSE_VERSION=1.7.0

# install dependencies
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
python3-dev python3-pip git g++ wget make libprotobuf-dev protobuf-compiler libopencv-dev \
libgoogle-glog-dev libboost-all-dev libcaffe-cuda-dev libhdf5-dev libatlas-base-dev

# install utils
RUN apt-get install -y unzip nano

# replace cmake, as old version has CUDA variable bugs
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.0/cmake-3.16.0-Linux-x86_64.tar.gz && \
tar xzf cmake-3.16.0-Linux-x86_64.tar.gz -C /opt && \
rm cmake-3.16.0-Linux-x86_64.tar.gz
ENV PATH="/opt/cmake-3.16.0-Linux-x86_64/bin:${PATH}"

#RUN distro=`lsb_release -sr`; \
#    comp=$(awk 'BEGIN{ print "'$distro'"<"'19.04'" }'); \
#    if [ "$comp" -eq 1 ]; then \
#        echo "WARNING: Getting newer CMake version from the kitware repository" ; \
#        wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - ;\
#        apt-add-repository "deb https://apt.kitware.com/ubuntu/ `lsb_release -cs` main";\
#        apt update ;\
#    fi
#RUN apt install -y cmake

# git clone openpose
WORKDIR /openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git .
    # && git checkout v$OPENPOSE_VERSION

# build openpose
WORKDIR /openpose/build
RUN cmake \
# bypass models download (sometimes the CMU server with the trained models does not work, blocking compilation)
# see https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1602#issuecomment-641653411
          -DDOWNLOAD_BODY_25_MODEL=OFF \
          -DDOWNLOAD_BODY_MPI_MODEL=OFF \
          -DDOWNLOAD_HAND_MODEL=OFF \
          -DDOWNLOAD_FACE_MODEL=OFF \
          ..

# remove compilation for the Ampere architecture (cuda 10 is not compatible with it, it needs cuda 11)
# see https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1753#issuecomment-792431838
RUN sed -ie 's/set(AMPERE "80 86")/#&/g'  ../cmake/Cuda.cmake && \
    sed -ie 's/set(AMPERE "80 86")/#&/g'  ../3rdparty/caffe/cmake/Cuda.cmake

RUN make -j`nproc`
RUN make install


######################
# set github ssh keys #
#######################

ARG ssh_prv_key
ARG ssh_pub_key

RUN apt-get update && \
apt-get install -y openssh-client git
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libmysqlclient-dev \
    ffmpeg libsm6 libxext6

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh
RUN ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_ed25519 && \
    echo "$ssh_pub_key" > /root/.ssh/id_ed25519.pub && \
    chmod 600 /root/.ssh/id_ed25519 && \
    chmod 600 /root/.ssh/id_ed25519.pub

############
#   YARP   #
############

RUN echo "*************** building yarp ****************"

ARG EVENT_DRIVEN_VERSION=1.5
ARG YARP_VERSION=3.4.4
ARG YCM_VERSION=0.12.1
ARG BUILD_TYPE=Release
ARG SOURCE_FOLDER=/usr/local
ARG OPENGL=0
ARG HPE_BRANCH=online_e2vid

ENV DEBIAN_FRONTEND noninteractive

RUN apt update

RUN apt install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common \
    lsb-core

# Install useful packages
RUN apt install -y \
        build-essential \
        libssl-dev

# Install yarp dependencies
RUN apt install -y \
        libgsl-dev \
        libedit-dev \
        libace-dev \
        libeigen3-dev \
# Install QT5 for GUIS
# (NOTE: may be not compatible with nvidia drivers when forwarding screen)
        qtbase5-dev \
        qt5-default \
        qtdeclarative5-dev \
        qtmultimedia5-dev \
        qml-module-qtquick2 \
        qml-module-qtquick-window2 \
        qml-module-qtmultimedia \
        qml-module-qtquick-dialogs \
        qml-module-qtquick-controls


RUN cd $SOURCE_FOLDER && \
    git clone https://github.com/robotology/ycm.git && \
    cd ycm && \
    git checkout v$YCM_VERSION && \
    mkdir build && cd build && \
    cmake .. && \
    make -j `nproc` install


# Install YARP
RUN cd $SOURCE_FOLDER && \
    git clone https://github.com/robotology/yarp.git &&\
    cd yarp && \
    git checkout v$YARP_VERSION && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
          -DYARP_COMPILE_BINDINGS=OFF \
          -DCREATE_PYTHON=OFF \
          .. && \
    make -j `nproc` install

RUN yarp check
EXPOSE 10000/tcp 10000/udp

# Some QT-Apps don't show controls without this
ENV QT_X11_NO_MITSHM 1


# Install event-driven
RUN cd $SOURCE_FOLDER && \
    git clone https://github.com/robotology/event-driven.git && \
    cd event-driven && \
    git checkout v$EVENT_DRIVEN_VERSION && \
    mkdir build && cd build && \
    cmake -DVLIB_CLOCK_PERIOD_NS=1000 \
          -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
          .. && \
    make install -j$(nproc)


#############
#   E2VID   #
#############

RUN echo "*************** installing e2vid ****************"

# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh && \
    chmod +x Anaconda3-2021.05-Linux-x86_64.sh && \
    bash Anaconda3-2021.05-Linux-x86_64.sh -b -p /root/anaconda && \
    rm Anaconda3-2021.05-Linux-x86_64.sh

# make non-activate conda commands available
ENV CONDA_DIR /root/anaconda
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --interative shells
RUN conda init

RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && \

#    conda create -n e2vid python=3.7 && \
#    conda activate e2vid && \
#    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch && \
#    conda install pandas && \
#    conda install -c conda-forge opencv && \
#    conda install -c anaconda scipy

    conda create -n e2vid && \
    conda activate e2vid && \
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch && \
    conda install pandas && \
    conda install -c anaconda scipy && \
    conda install -c conda-forge opencv

RUN echo "conda activate e2vid" >> ~/.bashrc

#ENV PYTHONHOME=/root/anaconda/bin
#ENV PYTHONPATH=/root/anaconda/lib:/root/anaconda/lib/python3.8:/root/anaconda/lib/python3.8/site-packages
#ENV PATH=$PYTHONHOME:$PATH

RUN cd $SOURCE_FOLDER && \
    git clone git@github.com:event-driven-robotics/hpe-core.git && \
    cd hpe-core && \
    git checkout $HPE_BRANCH

RUN cd hpe-core/evaluation/dhp19/e2vid && \
    git clone git@github.com:uzh-rpg/rpg_e2vid.git && \

    # fix e2vid incompatibility with recent pytorch versions (https://github.com/uzh-rpg/rpg_e2vid/issues/5) \
    cd rpg_e2vid && \
    sed -ie 's/index=xs[valid_indices]/index=(xs[valid_indices]/g' utils/inference_utils.py && \
    sed -ie 's/tis_long[valid_indices] * width * height/tis_long[valid_indices] * width * height).type(torch.cuda.LongTensor)/g' utils/inference_utils.py && \
    sed -ie 's/tis_long[valid_indices] + 1) * width * height/tis_long[valid_indices] + 1) * width * height).type(torch.cuda.LongTensor)/g' utils/inference_utils.py && \

    # download e2vid pretrained model
    wget "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" -O pretrained/E2VID_lightweight.pth.tar

WORKDIR hpe-core/evaluation/dhp19/e2vid/rpg_e2vid

# setup scripts for creating grayscale frames
RUN cp ../run_e2vid.py . && \
    cp ../../utils/mat_files.py . && \
    cp ../../utils/parsing.py . && \
    cp ../create_grayscale_frames.sh . && \
    chmod +x create_grayscale_frames.sh


#################
##   HPE-CORE   #
#################
#
#RUN echo "*************** building hpe-core ****************"
#
## Install hpe-core
#RUN cd $SOURCE_FOLDER && \
#    git clone git@github.com:event-driven-robotics/hpe-core.git && \
#    cd hpe-core && \
#    git checkout $HPE_VERSION && \
#    cd core && \
#    mkdir build && cd build && \
#    cmake .. && \
#    make install -j$(nproc) && \
#    cd ../.. && cd example/op_detector_example_module/ && \
#    mkdir build && cd build && \
#    cmake .. && \
#    make install -j$(nproc)
#
#
#
## add /usr/local/lib to the library path, so that libcaffe.so compiled with openpose will be used
## instead of the one provided by the nvidia/cuda docker image
#ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
#
#
######################
## OPENPOSE DOWNLOADS#
######################
#
## download models from google drive (https://drive.google.com/file/d/1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh/view)
## see https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/1602#issuecomment-641653411
#WORKDIR /openpose/tmp
#RUN git clone https://github.com/chentinghao/download_google_drive.git && \
#    pip3 install requests tqdm && \
#    cd download_google_drive && \
#    python3 download_gdrive.py 1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh ../../models.zip
#
## extract downloaded models
#WORKDIR /openpose
#RUN unzip models.zip && \
#    rm models.zip
#
#
#WORKDIR $SOURCE_FOLDER/hpe-core/example/op_detector_example_module/build

#- INCLUDE E2VID DEPENDENCIES IN DOCKER
#  - install e2vid vith anaconda
#  - make anaconda's python the default one that will be used from C++
#    $ export PYTHONHOME="/root/anaconda/bin"
#    $ export PYTHONPATH="/root/anaconda/lib:/root/anaconda/lib/python3.8:/root/anaconda/lib/python3.8/site-packages"
#    $ export PATH=$PYTHONHOME:$PATH

#- WRITE C++ YARP COMPONENT THAT CALLS PYTHON INTERPRETER
#  - print python version (must match anaconda's)

#- CREATE DATA STRUCTURE FOR READ EVENTS (NUMPY?)
#- PASS EVENTS TO E2VID
#- MAKE E2VID SEND FRAME TO OPENPOSE OR THE C++ YARP COMPONENT?