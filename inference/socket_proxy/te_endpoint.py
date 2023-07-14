import io
import requests

from flask import Flask
from flask_socketio import SocketIO,emit


app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*", path='tts_socket.io', async_handlers=True, pingTimeout=60000)
api_url = "http://localhost:5050"
# api_url = "https://tts-api.ai4bharat.org/"

@socketio.on('connect',namespace='/tts')
def connection(x):
    emit('connect','Connected tts')
    return 'connected'

@socketio.on('infer', namespace='/tts')
def infer(request_body):
    return requests.post(api_url, json=request_body).json()


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
