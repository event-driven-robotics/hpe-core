FROM ubuntu
LABEL maintainer="Gaurvi Goyal"

RUN apt-get update && apt-get install -y \
    python3-pip vim

COPY . /workspace

RUN pip install --upgrade pip && pip install -r workspace/requirements.txt

WORKDIR /workspace
