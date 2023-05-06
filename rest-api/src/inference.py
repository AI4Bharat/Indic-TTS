import io
import base64
import numpy as np
import traceback

from TTS.utils.synthesizer import Synthesizer
from aksharamukha.transliterate import process as aksharamukha_xlit
from scipy.io.wavfile import write as scipy_wav_write

import nltk
import enchant
from enchant.tokenize import get_tokenizer

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
        # TODO: Ability to instantiate models by accepting standard paths or auto-downloading
        
        if allow_transliteration:
            # Initialize Indic-Xlit models for the languages corresponding to TTS models
            from ai4bharat.transliteration import XlitEngine
            xlit_langs = set()
            
            for lang in list(models):
                if lang == 'en':
                    continue # No need of any Indic-transliteration for English
                
                if '+' in lang:
                    # If it's a code-mixed model like Hinglish, we need Hindi Xlit for non-English words
                    lang = lang.split('+')[1]
                xlit_langs.add(lang)
            
            self.xlit_engine = XlitEngine(xlit_langs, beam_width=6)
        else:
            self.xlit_engine = None

        self.text_normalizer = TextNormalizer()
        self.paragraph_handler = ParagraphHandler()

        self.orig_sr = 22050 # model.output_sample_rate
        self.enable_denoiser = enable_denoiser
        if enable_denoiser:
            from src.postprocessor import Denoiser
            self.target_sr = 16000
            self.denoiser = Denoiser(self.orig_sr, self.target_sr)
        else:
            self.target_sr = self.orig_sr
        
        self.post_processor = PostProcessor(self.target_sr)

        # Dictionary of English words
        self.enchant_dicts = {
            "en_US": enchant.Dict("en_US"),
            "en_GB": enchant.Dict("en_GB"),
        }
        self.enchant_tokenizer = get_tokenizer("en")

    def concatenate_chunks(self, wav: np.ndarray, wav_chunk: np.ndarray):
        # TODO: Move to utils
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

        # If there's no separate English model, use the Hinglish one
        if lang == "en" and lang not in self.models and "en+hi" in self.models:
            lang = "en+hi"

        if lang not in self.models:
            return TTSFailureResponse(status_text="Unsupported language!")
        
        if lang == "brx" and gender == "male":
            return TTSFailureResponse(status_text="Sorry, `male` speaker not supported for this language!")
        
        output_list = []

        for sentence in request.input:
            raw_audio = self.infer_from_text(sentence.source, lang, gender, transliterate_roman_to_native=transliterate_roman_to_native)
            # Convert PCM to WAV
            byte_io = io.BytesIO()
            scipy_wav_write(byte_io, self.target_sr, raw_audio)
            # Encode WAV fileobject as base64 for transmission via JSON
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
    ) -> np.ndarray:
        try:
            
            # If there's no separate English model, use the Hinglish one if present
            if lang == "en" and lang not in self.models and "en+hi" in self.models:
                lang = "en+hi"

            if lang == "en+hi": # Hinglish (English+Hindi code-mixed)
                primary_lang, secondary_lang = lang.split('+')
                input_text = self.transliterate_native_words_using_spell_checker(input_text, secondary_lang)
                # TODO: Write a proper `transliterate_native_words_using_eng_dictionary`
            else:
                primary_lang = lang
            
            input_text = self.text_normalizer.normalize_text(input_text, primary_lang)
            wav = None
            paragraphs = self.paragraph_handler.split_text(input_text)

            for paragraph in paragraphs:
                # Transliterate roman words to native script for Indic langs
                if transliterate_roman_to_native and primary_lang != 'en':
                    paragraph = self.transliterate_sentence(paragraph, primary_lang)
                
                # Manipuri was trained using the Central-govt's Bangla script
                # So convert the words in native state-govt script to Eastern-Nagari
                if primary_lang == "mni":
                    # TODO: Delete explicit-schwa
                    paragraph = aksharamukha_xlit("MeeteiMayek", "Bengali", paragraph)
                
                # Run Inference. TODO: Support for batch inference
                wav_chunk = self.models[lang].tts(paragraph, speaker_name=speaker_name, style_wav="")

                if self.enable_denoiser:
                    wav_chunk = self.denoiser.denoise(wav_chunk)
                wav_chunk = self.post_processor.process(wav_chunk, primary_lang, speaker_name)

                # Concatenate current chunk with previous audio outputs
                wav = self.concatenate_chunks(wav, wav_chunk)
            
            return wav
        except:
            traceback.print_exc()
            return np.zeros(1)

    def transliterate_native_words_using_spell_checker(self, input_text, lang):
        tokens = [result[0] for result in self.enchant_tokenizer(input_text)]
        pos_tags = [result[1] for result in nltk.tag.pos_tag(tokens)]

        # Transliterate non-English Roman words to Indic
        for word, pos_tag in zip(tokens, pos_tags):
            if pos_tag == "NNP" or pos_tag == "NNPS":
                # Enchant has many proper-nouns as well in its dictionary, don't know why.
                # So if it's a proper-noun, always nativize
                # FIXME: But NLTK's `averaged_perceptron_tagger` does not seem to be 100% accurate, it has false positives 🤦‍♂️ 
                pass
            elif self.enchant_dicts["en_US"].check(word) or self.enchant_dicts["en_GB"].check(word):
                # TODO: Merge British and American dicts into 1 somehow
                continue
            
            # Convert "Ram's" -> "Ram". TODO: Think what are the failure cases
            word = word.split("'")[0]

            transliterated_word = self.transliterate_sentence(word, lang)
            input_text = input_text.replace(word, transliterated_word, 1)
        return input_text

    def transliterate_sentence(self, input_text, lang):
        if not self.xlit_engine:
            return input_text

        if lang == "raj":
            lang = "hi" # Approximate
        
        return self.xlit_engine.translit_sentence(input_text, lang)
