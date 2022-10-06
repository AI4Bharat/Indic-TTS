import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from TTS.utils.synthesizer import Synthesizer

from src.inference import infer_from_request
from src.models.request import TTSRequest
from src.models.response import TTSFailureResponse

SUPPORTED_LANGUAGES = {
    "hi": "Hindi - हिंदी",
    "mr": "Marathi - मराठी",
}

models = {}
for lang in SUPPORTED_LANGUAGES:
    models[lang] = {}
    models[lang]['synthesizer']  = Synthesizer(
        tts_checkpoint=f'models/fastpitch/{lang}/male_female/best_model.pth',
        tts_config_path=f'models/fastpitch/{lang}/male_female/config.json',
        tts_speakers_file=f'models/fastpitch/{lang}/male_female/speakers.pth',
        # tts_speakers_file=None,
        tts_languages_file=None,
        vocoder_checkpoint=f'models/hifigan/{lang}/male_female/best_model.pth',
        vocoder_config=f'models/hifigan/{lang}/male_female/config.json',
        encoder_checkpoint="",
        encoder_config="",
        use_cuda=True,
    )
    print(f"Synthesizer loaded for {lang}.")
    print("*"*100)

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
    lang = request.config.language.sourceLanguage

    if lang not in SUPPORTED_LANGUAGES:
        return TTSFailureResponse(status_text="Unsupported language!")
    
    model = models[lang]['synthesizer']
    return infer_from_request(request, model)

if __name__ == "__main__":
    uvicorn.run("main:api", host="0.0.0.0", port=5050, log_level="info")
