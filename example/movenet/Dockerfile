FROM ubuntu
LABEL maintainer="Gaurvi Goyal"

RUN apt-get update && apt-get install -y \
    python3-pip

COPY ../. /examples
    
RUN pip install --upgrade pip && pip install -r examples/movenet/requirements.txt
