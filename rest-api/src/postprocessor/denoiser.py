import torch
import librosa
import numpy as np

class Denoiser:

    def __init__(self, orig_sr:int, target_sr:int):
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        from asteroid.models import BaseModel as AsteroidBaseModel
        self.model = AsteroidBaseModel.from_pretrained("JorisCos/DCCRNet_Libri1Mix_enhsingle_16k").to(self.device)
    
    def denoise(self, wav):
        if type(wav) != np.ndarray:
            wav = np.array(wav)
        
        if len(wav.shape) > 1:
            wav = np.mean(wav, axis=1)
        wav = librosa.resample(wav, orig_sr=self.orig_sr, target_sr=self.target_sr)
        wav = torch.Tensor(wav.reshape(1, 1, wav.shape[0])).float().to(self.device)
        wav = self.model.separate(wav)[0][0] #(batch, channels, time) -> (time)
        return wav.cpu().detach().numpy()
