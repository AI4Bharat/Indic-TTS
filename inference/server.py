import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from TTS.utils.synthesizer import Synthesizer

from src.inference import TextToSpeechEngine
from src.models.request import TTSRequest

SUPPORTED_LANGUAGES = {
    'as' : "Assamese - অসমীয়া",
    'bn' : "Bangla - বাংলা",
    'brx': "Boro - बड़ो",
    'en' : "English (Indian accent)",
    'en+hi': "English+Hindi (Hinglish code-mixed)",
    'gu' : "Gujarati - ગુજરાતી",
    'hi' : "Hindi - हिंदी",
    'kn' : "Kannada - ಕನ್ನಡ",
    'ml' : "Malayalam - മലയാളം",
    'mni': "Manipuri - মিতৈলোন",
    'mr' : "Marathi - मराठी",
    'or' : "Oriya - ଓଡ଼ିଆ",
    'pa' : "Panjabi - ਪੰਜਾਬੀ",
    'raj': "Rajasthani - राजस्थानी",
    'ta' : "Tamil - தமிழ்",
    'te' : "Telugu - తెలుగు",
}

models = {}
for lang in SUPPORTED_LANGUAGES:
    models[lang]  = Synthesizer(
        tts_checkpoint=f'checkpoints/{lang}/fastpitch/best_model.pth',
        tts_config_path=f'checkpoints/{lang}/fastpitch/config.json',
        tts_speakers_file=f'checkpoints/{lang}/fastpitch/speakers.pth',
        # tts_speakers_file=None,
        tts_languages_file=None,
        vocoder_checkpoint=f'checkpoints/{lang}/hifigan/best_model.pth',
        vocoder_config=f'checkpoints/{lang}/hifigan/config.json',
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=True,
    )
    print(f"Synthesizer loaded for {lang}.")
    print("*"*100)

engine = TextToSpeechEngine(models)

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/supported_languages")
def get_supported_languages():
    return SUPPORTED_LANGUAGES

@api.get("/")
def homepage():
    return "AI4Bharat Text-To-Speech API"

@api.post("/")
async def batch_tts(request: TTSRequest, response: Response):
    return engine.infer_from_request(request)

if __name__ == "__main__":
    # uvicorn server:api --host 0.0.0.0 --port 5050 --log-level info
    uvicorn.run("server:api", host="0.0.0.0", port=5050, log_level="info")
