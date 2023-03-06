import io
import base64
import numpy as np

from TTS.utils.synthesizer import Synthesizer
from aksharamukha.transliterate import process as aksharamukha_xlit
from scipy.io.wavfile import write as scipy_wav_write


from .models.common import Language
from .models.request import TTSRequest
from .models.response import AudioFile, AudioConfig, TTSResponse, TTSFailureResponse
from .utils.text import TextNormalizer
from .utils.paragraph_handler import ParagraphHandler

from src.postprocessor import PostProcessor

class TextToSpeechEngine:
    def __init__(
        self,
        models: dict,
        allow_transliteration: bool = True,
        enable_denoiser: bool = True,
    ):
        self.models = models
        if allow_transliteration:
            from ai4bharat.transliteration import XlitEngine
            self.xlit_engine = XlitEngine(list(models), beam_width=6)

        self.text_normalizer = TextNormalizer()
        self.orig_sr = 22050 # model.output_sample_rate
        self.enable_denoiser = enable_denoiser
        if enable_denoiser:
            self.target_sr = 16000
            self.post_processor = PostProcessor(self.orig_sr, self.target_sr)
        else:
            self.target_sr = self.orig_sr
        
        self.paragraph_handler = ParagraphHandler()

    def concatenate_chunks(self, wav: np.ndarray, wav_chunk: np.ndarray):
        if type(wav_chunk) != np.ndarray:
            wav_chunk = np.array(wav_chunk)
        if wav is None:
            return wav_chunk
        return np.concatenate([wav, wav_chunk])

    def infer_from_request(
        self,
        request: TTSRequest,
        transliterate_roman_to_native: bool = True
    ) -> TTSResponse:

        config = request.config
        lang = config.language.sourceLanguage
        gender = config.gender

        if lang not in self.models:
            return TTSFailureResponse(status_text="Unsupported language!")
        
        if lang == "brx" and gender == "male":
            return TTSFailureResponse(status_text="Sorry, `male` speaker not supported for this language!")
        
        output_list = []

        for sentence in request.input:
            wav = self.infer_from_text(sentence.source, lang, gender, transliterate_roman_to_native=transliterate_roman_to_native)

            byte_io = io.BytesIO()
            scipy_wav_write(byte_io, self.target_sr, wav)
            # model.save_wav(wav, byte_io)
            encoded_bytes = base64.b64encode(byte_io.read())
            encoded_string = encoded_bytes.decode()
            speech_response = AudioFile(audioContent=encoded_string)
            
            output_list.append(speech_response)

        audio_config = AudioConfig(language=Language(sourceLanguage=lang))
        return TTSResponse(audio=output_list, config=audio_config)
    
    def infer_from_text(
        self,
        input_text: str,
        lang: str,
        speaker_name: str,
        transliterate_roman_to_native: bool = True
    ):
        
        input_text = self.text_normalizer.normalize_text(input_text, lang)

        wav = None
        paragraphs = self.paragraph_handler.split_text(input_text)
        for paragraph in paragraphs:
            if transliterate_roman_to_native and lang != 'en':
                paragraph = self.transliterate_sentence(paragraph, lang)
            if lang == "mni":
                # TODO: Delete explicit-schwa
                paragraph = aksharamukha_xlit("MeeteiMayek", "Bengali", paragraph)               
            wav_chunk = self.models[lang].tts(paragraph, speaker_name=speaker_name, style_wav="")
            print("wav_chunk", type(wav_chunk))
            if self.enable_denoiser:
                wav_chunk = self.post_processor.process(wav_chunk, lang, speaker_name)
            wav = self.concatenate_chunks(wav, wav_chunk)
        return wav

    def transliterate_sentence(self, input_text, lang):
        if lang == "raj":
            lang = "hi" # Approximate
        
        return self.xlit_engine.translit_sentence(input_text, lang)
