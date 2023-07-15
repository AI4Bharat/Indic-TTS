import os
import sys
import io
import json
import tempfile

from TTS.utils.synthesizer import Synthesizer
import numpy as np

import triton_python_backend_utils as pb_utils

ENABLE_XLIT = True
INFERENCE_MODULE_DIR = "/home/app"
sys.path.insert(0, INFERENCE_MODULE_DIR)
from src.inference import TextToSpeechEngine

PWD = os.path.dirname(__file__)

class TritonPythonModel:

  def initialize(self, args):
    """`initialize` is called only once when the model is being loaded.
    Implementing `initialize` function is optional. This function allows
    the model to intialize any state associated with this model.
    Parameters
    ----------
    args : dict
     Both keys and values are strings. The dictionary keys and values are:
     * model_config: A JSON string containing the model configuration
     * model_instance_kind: A string containing model instance kind
     * model_instance_device_id: A string containing model instance device ID
     * model_repository: Model repository path
     * model_version: Model version
     * model_name: Model name
    """
     
    # You must parse model_config. JSON string is not parsed here
    self.model_config = model_config = json.loads(args['model_config'])
     
    self.model_instance_device_id = json.loads(args['model_instance_device_id'])

    # checkpoints_root_dir = os.path.join(PWD, "checkpoints")
    checkpoints_root_dir = "/models/checkpoints"
    checkpoint_folders = [ f.path for f in os.scandir(checkpoints_root_dir) if f.is_dir() ]
    # The assumption is that, each folder name is language code

    self.supported_speaker_ids = {"male", "female"}
    self.supported_lang_codes = set()
    self.models = {}

    for checkpoint_folder in checkpoint_folders:
      lang_code = os.path.basename(checkpoint_folder)

      # Replace a few hardcoded paths in the config
      tts_config_path = os.path.join(checkpoint_folder, "fastpitch/config.json")
      tts_config = json.load(open(tts_config_path))
      speakers_file = tts_config_path.replace("config.json", "speakers.pth")
      tts_config["model_args"]["speakers_file"] = speakers_file
      tts_config["speakers_file"] = speakers_file

      # Write the config file to a temporary path so that we can pass it to the Synthesizer class
      patched_tts_config_file = tempfile.NamedTemporaryFile(suffix=".json", mode='w', encoding='utf-8', delete=False)
      patched_tts_config_file.write(json.dumps(tts_config))
      patched_tts_config_file.close()

      self.models[lang_code] = Synthesizer(
        tts_checkpoint=os.path.join(checkpoint_folder, "fastpitch/best_model.pth"),
        tts_config_path=patched_tts_config_file.name,
        vocoder_checkpoint=os.path.join(checkpoint_folder, "hifigan/best_model.pth"),
        vocoder_config=os.path.join(checkpoint_folder, "hifigan/config.json"),
        use_cuda=True,
      )
      self.supported_lang_codes.add(lang_code)
      os.unlink(patched_tts_config_file.name)
    
    if "en+hi" in self.supported_lang_codes and "en" not in self.supported_lang_codes:
      self.supported_lang_codes.add("en")
    
    self.engine = TextToSpeechEngine(
      self.models,
      allow_transliteration=ENABLE_XLIT,
      enable_denoiser=False,
    )
     
  def execute(self,requests):
    responses = []

    for request in requests:
      
      input_texts = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT").as_numpy()
      speaker_ids = pb_utils.get_input_tensor_by_name(request, "INPUT_SPEAKER_ID").as_numpy()
      lang_ids = pb_utils.get_input_tensor_by_name(request, "INPUT_LANGUAGE_ID").as_numpy()
       
      input_texts = [input_text.decode("utf-8", "ignore") for input_text in input_texts]
      speaker_ids = [speaker_id.decode("utf-8", "ignore") for speaker_id in speaker_ids]
      lang_ids = [lang_id.decode("utf-8", "ignore") for lang_id in lang_ids]

      generated_audios = []
      
      for input_text, speaker_id, lang_id in zip(input_texts, speaker_ids, lang_ids):
        if lang_id in self.supported_lang_codes and speaker_id in self.supported_speaker_ids:
          # generated_audio = self.engine.models[lang_id].tts(input_text, speaker_id)
          generated_audio = self.engine.infer_from_text(input_text, lang=lang_id, speaker_name=speaker_id, transliterate_roman_to_native=ENABLE_XLIT)
        else:
          generated_audio = [0]
         
        generated_audios.append(generated_audio)
      
      out_tensor_0 = pb_utils.Tensor("OUTPUT_GENERATED_AUDIO",
                      np.array(generated_audios, dtype=np.float32))

       
      inference_response = pb_utils.InferenceResponse(
        output_tensors=[out_tensor_0])
      responses.append(inference_response)

    return responses
