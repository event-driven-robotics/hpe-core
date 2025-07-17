FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

ARG CODE_DIR=/usr/local/src

RUN apt update

# Add User ID and Group ID
ARG UNAME=vicon
ARG UID=1000
ARG GID=1000
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

# Add User into sudoers, can run sudo command without password
RUN apt update && apt install -y sudo
RUN usermod -aG sudo ${UNAME}
RUN echo "${UNAME} ALL=(ALL) NOPASSWD:ALL" | tee /etc/sudoers.d/${UNAME}

#basic environment
RUN apt install -y \
    ca-certificates \
    build-essential \
    git \
    cmake \
    cmake-curses-gui \
    libace-dev \
    libassimp-dev \
    libglew-dev \
    libglfw3-dev \
    libglm-dev \
    libeigen3-dev \
    clang-format \
    curl

# Install python environment
RUN apt update && apt install -y python3-dev python3-tk

# Install qt environment
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libxkbcommon-x11-0 \
    libxcb1 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libx11-xcb1 \
    libglu1-mesa \
    libxrender1 \
    libxi6 \
    libsm6 \
    libxext6 \
    libxrandr2 \
    libfontconfig1

# Change owner of some folders for the development
RUN chown -R $UNAME:$UNAME $CODE_DIR
RUN mkdir -p /app && chown -R $UNAME:$UNAME /app

USER $UNAME
WORKDIR /home/${UNAME}

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh