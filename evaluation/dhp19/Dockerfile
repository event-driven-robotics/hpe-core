
# base image
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# install utils
RUN apt-get update && \
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    wget \
    openssh-server \
    libmysqlclient-dev

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


###############
# setup e2vid #
###############

# install anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh && \
    chmod +x Anaconda3-2021.05-Linux-x86_64.sh && \
    bash Anaconda3-2021.05-Linux-x86_64.sh -b -p $HOME/anaconda

RUN eval "$(/root/anaconda/bin/conda shell.bash hook)" && \
    conda create -n hpe-core && \
    conda activate hpe-core && \
    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch && \
    conda install -c conda-forge opencv && \
    conda install -c anaconda scipy

# download hpe-core repository
RUN git clone git@github.com:event-driven-robotics/hpe-core.git

# download e2vid repository
RUN cd hpe-core/evaluation/convert_DHP19 && \
    git clone git@github.com:uzh-rpg/rpg_e2vid.git

# fix e2vid incompatibility with recent pytorch versions (https://github.com/uzh-rpg/rpg_e2vid/issues/5) \
RUN cd rpg_e2vid && \
    sed -ie 's/index=xs[valid_indices]/index=(xs[valid_indices]/g' utils/inference_utils.py \
    sed -ie 's/tis_long[valid_indices] * width * height/tis_long[valid_indices] * width * height).type(torch.cuda.LongTensor)/g' utils/inference_utils.py \
    sed -ie 's/tis_long[valid_indices] + 1) * width * height/tis_long[valid_indices] + 1) * width * height).type(torch.cuda.LongTensor)/g' utils/inference_utils.py

# download e2vid pretrained model
RUN wget "http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar" -O pretrained/E2VID_lightweight.pth.tar


# create grayscale frames using e2vid
RUN cp ../run_e2vid.py . && \
    cp ../utils/mat_files.py . && \
    cp ../create_grayscale_frames.sh . && \
    chmod +x create_grayscale_frames.sh

WORKDIR hpe-core/evaluation/convert_DHP19/rpg_e2vid
