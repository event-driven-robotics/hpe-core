# base image
FROM ubuntu:latest

MAINTAINER gaurvi goyal <gaurvi.goyal.iit.it>

RUN apt-get update


ENV TZ=Europe/Rome
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


######################
# set github ssh keys #
#######################

ARG ssh_prv_key
ARG ssh_pub_key

RUN apt-get update && apt-get install -y openssh-client git
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libmysqlclient-dev \
    ffmpeg libsm6 libxext6 cmake

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh
RUN ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_ed25519 && \
    echo "$ssh_pub_key" > /root/.ssh/id_ed25519.pub && \
    chmod 600 /root/.ssh/id_ed25519 && \
    chmod 600 /root/.ssh/id_ed25519.pub


##########
# GL-HPE #
##########

RUN echo "*************** building dependencies for gl-hpe ****************"


ARG SOURCE_FOLDER=/usr/local
ARG GLHPE_BRANCH=load_new_object
ARG HPE_VERSION=cpp-tests
# GL-HPE requirements 

# # Directories

RUN mkdir $SOURCE_FOLDER/code

RUN mkdir $SOURCE_FOLDER/data

RUN mkdir $SOURCE_FOLDER/data/DHP19

RUN mkdir $SOURCE_FOLDER/data/checkpoint

WORKDIR $SOURCE_FOLDER/code/

# # Packages

RUN apt-get install -y --no-install-recommends python3.8 python3-pip git python3-dev
RUN apt-get install -y swig

RUN apt-get install -y python3-opencv

RUN pip install albumentations==1.0.3 \
h5py==3.1.0 \
scikit-image==0.18.2 \
scikit-learn==0.24.2 \
scikit-video==1.1.11 \
scipy==1.7.1 \
torch==1.9.0 \
kornia==0.5.7 \
hydra-core==1.1.0 \
omegaconf==2.1.0 \
opencv-python==4.5.3.56 \
pytorch-lightning==1.1.6 \
torchvision==0.10.0 \
tqdm==4.62.0 \
numpy==1.17.5 \
matplotlib==3.4.2 \
segmentation_models_pytorch==0.2.0 \
sklearn \
-e git+https://github.com/anibali/pose3d-utils#egg=pose3d_utils \
#-e git+https://github.com/IIT-PAVIS/event_library#egg=event_library \
#-e git+https://github.com/gaurvigoyal/gl_hpe@load_new_object#egg=gl_hpe \
comet_ml 

# NVIDIA INSTALLATIONS- TO TEST

# RUN sudo apt-get install nvidia-cuda-toolkit

WORKDIR $SOURCE_FOLDER/code/




# Install GL-HPE program
RUN git clone git@github.com:gaurvigoyal/gl_hpe.git
RUN cd gl_hpe && pwd && \
git status && git checkout load_new_object
ENV PYTHONPATH $PYTHONPATH:$SOURCE_FOLDER/code/gl_hpe



# download models from google drive (https://drive.google.com/file/d/1z1XLPWMVsTpSZnXP8ERXHcR0dss_dhwy/view?usp=sharing)

RUN cd $SOURCE_FOLDER/code/gl_hpe && \
    mkdir checkpoint && \
    cd checkpoint && \
    git clone git@github.com:chentinghao/download_google_drive.git && \
    pip install requests && \
    cd download_google_drive && \
    python3 download_gdrive.py 1z1XLPWMVsTpSZnXP8ERXHcR0dss_dhwy ../epoch=19-val_loss=0.09.ckpt && \
    cd .. && rm -r download_google_drive


############
#   YARP   #
############

RUN echo "*************** building yarp ****************"

ARG EVENT_DRIVEN_VERSION=1.5
ARG YARP_VERSION=3.4.3
ARG YCM_VERSION=0.13.0
ARG BUILD_TYPE=Release
ARG SOURCE_FOLDER=/usr/local
ARG OPENGL=0

ENV DEBIAN_FRONTEND noninteractive 

# RUN apt-get update

RUN apt-get install -y \
    apt-transport-https \
    ca-certificates \
    gnupg \
    software-properties-common \
    lsb-core
    
# Install useful packages
RUN apt-get install -y \
        build-essential \
        libssl-dev
        
# Install yarp dependencies
RUN apt-get install -y \
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

RUN sh -c 'echo "deb http://www.icub.org/ubuntu `lsb_release -cs` contrib/science" > /etc/apt/sources.list.d/icub.list'
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 57A5ACB6110576A6
RUN apt-get update
RUN apt install -y icub-common

RUN cd $SOURCE_FOLDER
RUN    git clone git@github.com:robotology/ycm.git && \
    cd ycm && \
    git checkout v$YCM_VERSION && \
    mkdir build && cd build && \
    cmake .. && \
    make -j `nproc` install


# Install YARP
RUN cd $SOURCE_FOLDER && \
    git clone git@github.com:robotology/yarp.git &&\
    cd yarp && \
    apt-get install -y python3-dev && \
    git checkout v$YARP_VERSION && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
          -DYARP_COMPILE_BINDINGS=ON \
          -DCREATE_PYTHON=ON \
          .. && \
    make -j `nproc` install

RUN yarp check
EXPOSE 10000/tcp 10000/udp

# Some QT-Apps don't show controls without this
ENV QT_X11_NO_MITSHM 1


# Install event-driven
RUN cd $SOURCE_FOLDER && \
    git clone git@github.com:robotology/event-driven.git && \
    cd event-driven && \
    git checkout v$EVENT_DRIVEN_VERSION && \
    mkdir build && cd build && \
    cmake -DVLIB_CLOCK_PERIOD_NS=1000 \
          -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
          .. && \
    make install -j$(nproc)

RUN echo "*************** building hpe-core ****************"

# Install hpe-core
RUN cd $SOURCE_FOLDER && \
    git clone git@github.com:event-driven-robotics/hpe-core.git && \
    cd hpe-core && \
    git checkout $HPE_VERSION && \
    cd core && \
    mkdir build && cd build && \
    cmake .. && \
    make install -j$(nproc) && \
    cd ../.. && cd example/yarp-glhpe/ && \
    mkdir build && cd build && \
    cmake .. && \
    make install -j$(nproc)

# add /usr/local/lib to the library path, so that libcaffe.so compiled with openpose will be used
# instead of the one provided by the nvidia/cuda docker image
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# Adding yarp python file to the pythonpath
ENV PYTHONPATH $PYTHONPATH:/usr/local/yarp/build/lib/python3





