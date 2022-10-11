import json
from flask import Flask
from flask_socketio import SocketIO,emit
import time
import traceback
import numpy as np

import io
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from TTS.utils.synthesizer import Synthesizer

# class Item(BaseModel):
#     text: str
#     lang: str | None = 'te'


langs = ['te',       # single-speaker
         'hi', 'mr'  # multi-speaker
        ]
models = {}
for lang in langs:
    speakers = 'male_female' if lang != 'te' else 'male'
    speakers_file = f'saved_models/fastpitch/{lang}/{speakers}/speakers.pth' if lang != 'te' else None
    models[lang] = {}
    models[lang]['synthesizer']  = Synthesizer(
        tts_checkpoint=f'saved_models/fastpitch/{lang}/{speakers}/best_model.pth',
        tts_config_path=f'saved_models/fastpitch/{lang}/{speakers}/config.json',
        tts_speakers_file=None,
        tts_languages_file=None,
        vocoder_checkpoint=f'saved_models/hifigan/{lang}/male_female/best_model.pth',
        vocoder_config=f'saved_models/hifigan/{lang}/male_female/config.json',
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=False,
    )
    print(f"Synthesizer loaded for {lang}.")
    print("*"*100)

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", path='tts_socket.io', async_handlers=True, pingTimeout=60000)

@socketio.on('connect',namespace='/tts')
def connection(x):
    emit('connect','Connected tts')
    return 'connected'

@socketio.on('infer', namespace='/tts')
def infer(req_data):
    status = "SUCCESS"
    start_time = time.time()
    text = req_data['text']
    language = req_data['language']
    if text in [None, ''] or language in [None, '']:
        status = 'ERROR'
        print(traceback.format_exc())
    speaker_name = req_data['speaker'] if language != 'te' else ""
    wavs = models[language]['synthesizer'].tts(text, speaker_name=speaker_name, style_wav="")
    out = io.BytesIO()
    models[language]['synthesizer'].save_wav(wavs, out)
    print(time.time() - start_time, "Total Time elapsed!")
    return {"status":status, "output": {'audio': out.read()}}


# @app.route("/endpoint", methods=["GET", "POST"])
# def endpoint_http():
#     return [1, 2, 3]


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
