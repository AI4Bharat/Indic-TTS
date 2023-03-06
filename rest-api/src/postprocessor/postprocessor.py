import os
import time
import torch
import ffmpeg
import librosa
import numpy as np
import soundfile as sf
from uuid import uuid4

from .vad import VoiceActivityDetection


class PostProcessor:

    def __init__(self, orig_sr:int, target_sr:int):
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        from asteroid.models import BaseModel as AsteroidBaseModel
        self.denoiser = AsteroidBaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k").to(self.device)
        self.vad = VoiceActivityDetection()
        os.makedirs('wavs', exist_ok=True)  # tmp directory for ffmpeg processing

    def set_tempo(self, wav:np.ndarray, atempo:str ='1'):
        inpath = 'wavs/' + str(uuid4()) + '.wav'
        outpath = inpath.replace('.wav', '_.wav')
        sf.write(inpath, wav, self.target_sr)
        in_stream = ffmpeg.input(inpath)
        audio_stream = ffmpeg.filter_(in_stream, 'atempo', atempo)
        audio_stream = audio_stream.output(outpath)
        ffmpeg.run(audio_stream, overwrite_output=True)
        wav, _ = librosa.load(outpath, sr=self.target_sr)
        os.remove(inpath)
        os.remove(outpath)
        return wav
    
    def trim_silence(self, wav:np.ndarray):
        return self.vad.process(wav, sc_threshold=40)
    
    def denoise(self, wav:np.ndarray):
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)
        wav = librosa.resample(wav, orig_sr=self.orig_sr, target_sr=self.target_sr)
        wav = torch.Tensor(wav.reshape(1, 1, wav.shape[0])).float().to(self.device)
        wav = self.denoiser.separate(wav)[0][0] #(batch, channels, time) -> (time)
        return wav.cpu().detach().numpy()

    def process(self, wav_obj:list, lang:str, gender:str):
        # st = time.time()
        wav = np.array(wav_obj)
        
        # Denoiser
        wav = self.denoise(wav)

        if (lang == "te") and (gender=='female'):  # Telugu female speaker slow down
            wav = self.set_tempo(wav, '0.85')
            wav = self.trim_silence(wav) 
        elif (lang == 'mr') and (gender=='female'):  # Marathi female speaker speed up
            wav = self.trim_silence(wav)
            wav = self.set_tempo(wav, '1.15')
        # et = time.time()
        # print('Elapsed: ', et - st, 's')
        return wav
