import tritonclient.http as http_client
from tritonclient.utils import *

ENDPOINT_URL = 'localhost:8000'
triton_http_client = http_client.InferenceServerClient(
    url=ENDPOINT_URL, verbose=False,
)

print("Is server ready - {}".format(triton_http_client.is_server_ready()))

import io
from scipy.io.wavfile import write as scipy_wav_write
import numpy as np

def get_string_tensor(string_value, tensor_name):
    string_obj = np.array([string_value], dtype="object")
    input_obj = http_client.InferInput(tensor_name, string_obj.shape, np_to_triton_dtype(string_obj.dtype))
    input_obj.set_data_from_numpy(string_obj)
    return input_obj

# inputs = [get_string_tensor("নমস্তে", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("as", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("নমস্তে", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("bn", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("નમસ્તે", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("gu", "INPUT_LANGUAGE_ID")]
inputs = [get_string_tensor("सलाम", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("hi", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("नमस्ते", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("mr", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("ନମସ୍ତେ", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("or", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("ਨਮਸਤੇ", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("pa", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("सलाम", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("raj", "INPUT_LANGUAGE_ID")]

# inputs = [get_string_tensor("ನಮಸ್ಕಾರಂ", "INPUT_TEXT"), get_string_tensor("male", "INPUT_SPEAKER_ID"), get_string_tensor("kn", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("നമസ്കാരം", "INPUT_TEXT"), get_string_tensor("male", "INPUT_SPEAKER_ID"), get_string_tensor("ml", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("வணக்கம்‌", "INPUT_TEXT"), get_string_tensor("male", "INPUT_SPEAKER_ID"), get_string_tensor("ta", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("నమస్కారం", "INPUT_TEXT"), get_string_tensor("male", "INPUT_SPEAKER_ID"), get_string_tensor("te", "INPUT_LANGUAGE_ID")]

# inputs = [get_string_tensor("नमस्ते", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("brx", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("নমস্তে", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("mni", "INPUT_LANGUAGE_ID")]
# inputs = [get_string_tensor("Greetings", "INPUT_TEXT"), get_string_tensor("female", "INPUT_SPEAKER_ID"), get_string_tensor("en", "INPUT_LANGUAGE_ID")]

output0 = http_client.InferRequestedOutput("OUTPUT_GENERATED_AUDIO")

response = triton_http_client.infer("tts", model_version='1', inputs=inputs, outputs=[output0])#.get_response()
wav = response.as_numpy("OUTPUT_GENERATED_AUDIO")[0]
byte_io = io.BytesIO()
scipy_wav_write(byte_io, 22050, wav)

with open("audio.wav", "wb") as f:
    f.write(byte_io.read())
