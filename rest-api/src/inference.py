import io
import torch
import base64
import ffmpeg
import librosa
import numpy as np
import soundfile as sf
from uuid import uuid4


from TTS.utils.synthesizer import Synthesizer
from asteroid.models import BaseModel as AsteroidBaseModel
from aksharamukha.transliterate import process as aksharamukha_xlit
from scipy.io.wavfile import write as scipy_wav_write


from src.models.common import Language
from src.models.request import TTSRequest
from src.models.response import AudioFile, AudioConfig, TTSResponse, TTSFailureResponse

class TextToSpeechEngine:
    def __init__(self, models: dict, allow_transliteration=True):
        self.models = models
        if allow_transliteration:
            from ai4bharat.transliteration import XlitEngine
            self.xlit_engine = XlitEngine(list(models), beam_width=6)

        self.orig_sr = 22050
        self.target_sr = 16000
        self.postprocessor = AsteroidBaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k")
        

    def infer_from_request(self, request: TTSRequest, transliterate_roman_to_indic: bool = True):
        config = request.config
        lang = config.language.sourceLanguage
        gender = config.gender

        if lang not in self.models:
            return TTSFailureResponse(status_text="Unsupported language!")
        
        if lang == "mni" and gender == "male":
            return TTSFailureResponse(status_text="Sorry, `male` speaker not supported for this language!")
        
        model = self.models[lang]['synthesizer']
        output_list = []

        for sentence in request.input:
            input_text = sentence.source
            if transliterate_roman_to_indic:
                input_text = self.transliterate_sentence(input_text, lang)
            
            if lang == "mni":
                # TODO: Delete explicit-schwa
                input_text = aksharamukha_xlit("MeeteiMayek", "Bengali", input_text)
            
            wav_obj = model.tts(input_text, speaker_name=gender, style_wav="")
            
            # Denoiser
            wav = np.array(wav_obj)
            if len(wav.shape) > 1:
                wav = np.mean(wav, axis=1)
            wav = librosa.resample(wav, orig_sr=self.orig_sr, target_sr=self.target_sr)
            wav = torch.Tensor(wav.reshape(1, 1, wav.shape[0])).float()
            wav = self.postprocessor.separate(wav)[0][0] #(batch, channels, time) -> (time)


            byte_io = io.BytesIO()
            scipy_wav_write(byte_io, self.target_sr, wav.cpu().detach().numpy())
            # model.save_wav(wav, byte_io)
            encoded_bytes = base64.b64encode(byte_io.read())
            encoded_string = encoded_bytes.decode()
            speech_response = AudioFile(audioContent=encoded_string)
            
            output_list.append(speech_response)

        audio_config = AudioConfig(language=Language(sourceLanguage=lang))
        return TTSResponse(audio=output_list, config=audio_config)

    def transliterate_sentence(self, input_text, lang):
        if lang == "raj":
            lang = "hi" # Approximate
        
        return self.xlit_engine.translit_sentence(input_text, lang)
