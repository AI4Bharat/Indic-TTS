ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:22.12-py3
FROM ${BASE_IMAGE}

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8

RUN apt-get update && apt-get install --fix-missing -y libsndfile1-dev ffmpeg enchant

WORKDIR /models
COPY checkpoints ./checkpoints

WORKDIR /home/app

RUN git clone https://github.com/gokulkarthik/TTS 
RUN cd TTS && python3 -m pip install .[all]

RUN python3 -m pip install pyworld==0.3.1 aksharamukha nltk pydantic nemo_text_processing indic_numtowords indic-nlp-library translators librosa numpy ffmpeg-python soundfile

RUN git clone https://github.com/AI4Bharat/IndicXlit.git
RUN cd IndicXlit/app && python3 -m pip install .

COPY src ./src

WORKDIR /home
COPY triton_server/triton_repo ./triton_repo

CMD ["tritonserver", "--model-repository=/home/triton_repo", "--log-verbose=2", "--strict-model-config=false", "--http-port=8000", "--grpc-port=8001", "--metrics-port=8002"]
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002
