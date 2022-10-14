import io
import base64

from TTS.utils.synthesizer import Synthesizer
from aksharamukha.transliterate import process as aksharamukha_xlit

from src.models.common import Language
from src.models.request import TTSRequest
from src.models.response import AudioFile, AudioConfig, TTSResponse, TTSFailureResponse

class TextToSpeechEngine:
    def __init__(self, models: dict, allow_transliteration=True):
        self.models = models
        if allow_transliteration:
            from ai4bharat.transliteration import XlitEngine
            self.xlit_engine = XlitEngine(list(models), beam_width=6)
        

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
            byte_io = io.BytesIO()
            model.save_wav(wav_obj, byte_io)
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
