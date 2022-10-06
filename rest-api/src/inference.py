import io
import base64

from TTS.utils.synthesizer import Synthesizer

from src.models.common import Language
from src.models.request import TTSRequest
from src.models.response import AudioFile, AudioConfig, TTSResponse, TTSFailureResponse

def infer_from_request(request: TTSRequest, model: Synthesizer):
    config = request.config
    lang = config.language.sourceLanguage
    gender = config.gender
    
    output_list = []

    for sentence in request.input:
        wav_obj = model.tts(sentence.source, speaker_name=gender, style_wav="")
        byte_io = io.BytesIO()
        model.save_wav(wav_obj, byte_io)
        encoded_bytes = base64.b64encode(byte_io.read())
        encoded_string = encoded_bytes.decode()
        speech_response = AudioFile(audioContent=encoded_string)
        
        output_list.append(speech_response)

    audio_config = AudioConfig(language=Language(sourceLanguage=lang))
    return TTSResponse(audio=output_list, config=audio_config)
