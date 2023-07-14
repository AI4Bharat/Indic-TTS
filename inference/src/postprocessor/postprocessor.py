import os
import ffmpeg
import librosa
import numpy as np
import soundfile as sf
import tempfile

from .vad import VoiceActivityDetection


class PostProcessor:

    def __init__(self, target_sr:int):
        self.target_sr = target_sr
        self.vad = VoiceActivityDetection()

    def set_tempo(self, wav:np.ndarray, atempo:str ='1'):
        with tempfile.TemporaryDirectory() as tmpdirname:
            inpath = os.path.join(tmpdirname, 'input.wav')
            outpath = inpath.replace('input.wav', 'output.wav')
            sf.write(inpath, wav, self.target_sr)
            in_stream = ffmpeg.input(inpath)
            audio_stream = ffmpeg.filter_(in_stream, 'atempo', atempo)
            audio_stream = audio_stream.output(outpath)
            ffmpeg.run(audio_stream, overwrite_output=True)
            wav, _ = librosa.load(outpath, sr=self.target_sr)
        return wav
    
    def trim_silence(self, wav:np.ndarray):
        return self.vad.process(wav, sc_threshold=40)

    def process(self, wav, lang:str, gender:str):
        if type(wav) != np.ndarray:
            wav = np.array(wav)

        if (lang == "te") and (gender=='female'):  # Telugu female speaker slow down
            wav = self.set_tempo(wav, '0.85')
            wav = self.trim_silence(wav) 
        elif (lang == 'mr') and (gender=='female'):  # Marathi female speaker speed up
            wav = self.trim_silence(wav)
            wav = self.set_tempo(wav, '1.15')
        elif (lang == 'gu'):  # Gujarati speaker speed up
            # wav = trim_silence(wav)
            wav = self.set_tempo(wav, '1.20')

        return wav
