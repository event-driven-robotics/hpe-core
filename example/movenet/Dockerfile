
# base image
FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# install utils
RUN apt-get update && \
    apt-get install -y unzip nano git wget sudo build-essential libssl-dev cmake

###########################
# CUDA PUBLIC KEYS UPDATE #
###########################

RUN sudo apt-key del 7fa2af80
RUN sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

##########
# PYTHON #
##########


# update python
ARG PYTHON_VERSION=3.8
RUN apt-get update &&  \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y libcanberra-gtk-module mesa-utils python$PYTHON_VERSION python3-pip python3-tk

# create list of alternative python interpreters
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python$PYTHON_VERSION 1 && \
    update-alternatives --config python3 && \
    rm /usr/bin/python3 && \
    ln -s python$PYTHON_VERSION /usr/bin/python3

# upgrade pip
RUN python3 -m pip install --upgrade pip

# other prerequisites
RUN apt-get install -y libopencv-dev python3-opencv python3-dev




#######################
# set github ssh keys #
#######################

ARG ssh_prv_key
ARG ssh_pub_key

RUN apt-get update && \
apt-get install -y openssh-client git
RUN apt-get install -y \
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


###############
# ATIS BRIDGE #
###############

RUN if [ $hvga -eq 1 ]; then \
		echo "Installing older 1.4 version for HVGA"; \
		echo "deb [arch=amd64 trusted=yes] https://prophesee:DbnLdKL5YXnMndWg@apt.prophesee.ai/dists/ubuntu focal main" >> /etc/apt/sources.list; \
    apt update; \
    apt install -y prophesee-* ; \
	else \
    	echo "deb [arch=amd64 trusted=yes] https://apt.prophesee.ai/dists/public/7l58osgr/ubuntu focal essentials" >> /etc/apt/sources.list; \
      apt update; \
      apt install -y metavision-*; \
    fi

RUN python3 -m pip install scikit-build matplotlib jupyter pandas

RUN apt install -y \
  libcanberra-gtk-module \
  python3.7 \
  python3-pip \
  python3-tk \
  cmake \
  && apt-get autoremove \
  && apt-get clean \
  && rm -rf /tmp/* /var/lib/apt/lists/* /var/tmp/*

RUN pip3 install \
  scikit-build

RUN pip3 install \
  matplotlib \
  jupyter


############
#   YARP   #
############

ARG EVENT_DRIVEN_VERSION=1.5
ARG YARP_VERSION=3.6.0
ARG YCM_VERSION=0.14.0
ARG BIMVEE_COMMIT=3670bcc9c75608df1cfdef06f3f1a34b0ca68d43
# mustard first release
ARG MUSTARD_COMMIT=18e702892f6028c69a2a9e224f459bc26147b20e
ARG BUILD_TYPE=Release
ARG SOURCE_FOLDER=/usr/local
ARG OPENGL=0
ARG HPE_BRANCH=main

RUN apt update

RUN apt install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-core \
    swig

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


# Install YCM
RUN cd $SOURCE_FOLDER && \
    git clone https://github.com/robotology/ycm.git && \
    cd ycm && \
    git checkout v$YCM_VERSION && \
#    # FIXME: remove this line after gitlab fix!!!!!
#    sed -ie 's/libeigen/cantonios/g' 3rdparty/eigen3.cmake && \
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
          -DYARP_COMPILE_BINDINGS=ON \
          -DCREATE_PYTHON=ON \
          .. && \
    make -j `nproc` install

RUN yarp check
EXPOSE 10000/tcp 10000/udp

# make yarp's python binding visible to python interpreter
ENV PYTHONPATH $SOURCE_FOLDER/yarp/build/lib/python3:$PYTHONPATH

# Some QT-Apps don't show controls without this
ENV QT_X11_NO_MITSHM 1


# install EVENT-DRIVEN
RUN cd $SOURCE_FOLDER && \
    git clone https://github.com/robotology/event-driven.git && \
    cd event-driven && \
    git checkout v$EVENT_DRIVEN_VERSION && \
    mkdir build && cd build && \
    cmake -DVLIB_CLOCK_PERIOD_NS=1000 \
          -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
          -DBUILD_HARDWAREIO=ON\
          -DENABLE_atis3=ON\
          .. && \
    make install -j$(nproc)


# install BIMVEE
RUN python3 -m pip install setuptools Cython numpy==1.21.4 scipy  # numpy==1.21.4 if using python3.7
RUN cd $SOURCE_FOLDER && \
    git clone https://github.com/event-driven-robotics/bimvee --recurse-submodules && \
    cd bimvee && \
    git checkout $BIMVEE_COMMIT && \
    python3 setup.py install


# install MUSTARD
RUN cd $SOURCE_FOLDER && \
    git clone https://github.com/event-driven-robotics/mustard && \
    cd mustard && \
    git checkout $MUSTARD_COMMIT && \
    python3 setup.py install



############
# HPE-CORE #
############

# download hpe-core
RUN cd $SOURCE_FOLDER && \
    git clone git@github.com:event-driven-robotics/hpe-core.git && \
    cd hpe-core && \
    git checkout $HPE_BRANCH

# compile and install hpe-core
RUN cd $SOURCE_FOLDER/hpe-core/core && \
    mkdir build && cd build && \
    cmake .. && \
    make install -j$(nproc)

# install movenet dependencies
RUN cd $SOURCE_FOLDER/hpe-core/example/movenet && \
    git checkout movenet_integration && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    mkdir build && cd build && \
    cmake .. && make install -j$(nproc)

RUN export PYTHONPATH=$PYTHONPATH:$SOURCE_FOLDER/hpe-core

WORKDIR $SOURCE_FOLDER/hpe-core/example/movenet
