ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.12-py3
FROM ${BASE_IMAGE}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8

RUN apt-get update && apt-get install --fix-missing -y libsndfile1-dev ffmpeg enchant

WORKDIR /home/app

COPY requirements-server.txt .
RUN pip install --use-deprecated=legacy-resolver -r requirements-server.txt
COPY requirements-ml.txt .
RUN pip install --use-deprecated=legacy-resolver -r requirements-ml.txt
COPY requirements-utils.txt .
RUN pip install --use-deprecated=legacy-resolver -r requirements-utils.txt

# Download depedencies
COPY examples ./examples
RUN python3 examples/xlit.py
RUN python3 examples/pos_tag.py

COPY src ./src

WORKDIR /home
COPY triton_server/triton_repo ./triton_repo

CMD ["tritonserver", "--model-repository=/home/triton_repo", "--log-verbose=2", "--strict-model-config=false", "--http-port=8000", "--grpc-port=8001", "--metrics-port=8002"]
EXPOSE 8000
EXPOSE 8001 
EXPOSE 8002
