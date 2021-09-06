
# base image
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# install utils
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    vim \
    wget \
    openssh-server \
    libmysqlclient-dev \
    ffmpeg libsm6 libxext6

#######################
# set github ssh keys #
#######################
ARG ssh_prv_key
ARG ssh_pub_key

# Authorize SSH Host
RUN mkdir -p /root/.ssh && \
    chmod 0700 /root/.ssh && \
    ssh-keyscan github.com > /root/.ssh/known_hosts

# Add the keys and set permissions
RUN echo "$ssh_prv_key" > /root/.ssh/id_rsa && \
    echo "$ssh_pub_key" > /root/.ssh/id_rsa.pub && \
    chmod 600 /root/.ssh/id_rsa && \
    chmod 600 /root/.ssh/id_rsa.pub


##################
# setup anaconda #
##################

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
    conda create -n hpe-core && \
    conda activate hpe-core && \
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch && \
    conda install pandas && \
    conda install -c conda-forge opencv && \
    conda install -c anaconda scipy

RUN echo "conda activate hpe-core" >> ~/.bashrc


###############
# setup e2vid #
###############

# download hpe-core repository
RUN git clone git@github.com:event-driven-robotics/hpe-core.git && \
    cd hpe-core && \
    git checkout docker_e2vid

# download e2vid repository
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
