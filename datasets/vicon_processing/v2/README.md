## Docker and Virtual Environment setup
### Build docker image
```
## host
XXX@YYY:~/Github/hpe-core $ docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -f datasets/vicon_processing/v2/Dockerfile -t vicon:latest .
```
### Run and enter docker container with docker compose
```
## Host
XXX@YYY:~/Github/hpe-core $ docker compose -f datasets/vicon_processing/v2/docker-compose.yaml up -d
docker exec -it vicon /bin/bash
```

### Create python environment with `uv`
`uv` is an extremely fast Python package and project manager, written in Rust. You can refer to formal documents in this [[link](https://docs.astral.sh/uv/getting-started/installation/)]. <br>
```
## Docker container
cd /app/hpe-core/datasets/vicon_processing/v2
uv sync
```