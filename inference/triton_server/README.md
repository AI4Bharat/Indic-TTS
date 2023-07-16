# Triton server

## Building the image

```
cd inference/
docker build -f triton_server/Dockerfile -t tts_triton .
```

## Running the container

Then start the server by:
```
docker run --shm-size=256m --gpus=1 --rm -v ${PWD}/checkpoints/:/models/checkpoints -p 8000:8000 -t tts_triton
```

## Sample client

- Do `pip install tritonclient gevent` first.
- Then `python3 triton_server/client.py`, which will generate `audio.wav`
