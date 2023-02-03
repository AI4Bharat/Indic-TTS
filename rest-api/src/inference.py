import io
import base64


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
    def __init__(self, models: dict, allow_transliteration=True):
        self.models = models
        if allow_transliteration:
            from ai4bharat.transliteration import XlitEngine
            self.xlit_engine = XlitEngine(list(models), beam_width=6)

        self.text_normalizer = TextNormalizer()
        self.orig_sr = 22050
        self.target_sr = 16000
        self.post_processor = PostProcessor(self.orig_sr, self.target_sr)
        self.paragraph_handler = ParagraphHandler()

    def infer_from_request(self, request: TTSRequest, transliterate_roman_to_indic: bool = True):
        config = request.config
        lang = config.language.sourceLanguage
        gender = config.gender

        if lang not in self.models:
            return TTSFailureResponse(status_text="Unsupported language!")
        
        if lang == "brx" and gender == "male":
            return TTSFailureResponse(status_text="Sorry, `male` speaker not supported for this language!")
        
        model = self.models[lang]['synthesizer']
        output_list = []

        for sentence in request.input:
            input_text = sentence.source
            print('input:', input_text)
            print("Normalizing")
            input_text = self.text_normalizer.normalize_text(input_text, lang)
            # print("roman2indic")
            # if transliterate_roman_to_indic and lang != 'en':
                # input_text = self.transliterate_sentence(input_text, lang)
            # if lang == "mni":
                # TODO: Delete explicit-schwa
                # input_text = aksharamukha_xlit("MeeteiMayek", "Bengali", input_text)
            print("Split para")
            wav = None
            paragraphs = self.paragraph_handler.split_text(input_text)
            for paragraph in paragraphs:
                print("para:", paragraph)
                print("roman2indic")
                if transliterate_roman_to_indic:
                    paragraph = self.transliterate_sentence(paragraph, lang)
                    print('translit text', paragraph) 
                if lang == "mni":
                    # TODO: Delete explicit-schwa
                    paragraph = aksharamukha_xlit("MeeteiMayek", "Bengali", paragraph)               
                wav_obj = model.tts(paragraph, speaker_name=gender, style_wav="")
                wav_chunk = self.post_processor.process(wav_obj, lang, gender)
                wav = self.post_processor.concatenate_chunks(wav, wav_chunk)

            print("Creating payload")
            byte_io = io.BytesIO()
            scipy_wav_write(byte_io, self.target_sr, wav)
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
