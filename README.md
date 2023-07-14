# AI4Bharat Indic-TTS

## Towards Building Text-To-Speech Systems for the Next Billion Users

> 🎉 Accepted at ICASSP 2023

Deep learning based text-to-speech (TTS) systems have been evolving rapidly with advances in model architectures, training methodologies, and generalization across speakers and languages. However, these advances have not been thoroughly investigated for Indian language speech synthesis. Such investigation is computationally expensive given the number and diversity of Indian languages, relatively lower resource availability, and the diverse set of advances in neural TTS that remain untested.  In this paper, we evaluate the choice of acoustic models, vocoders, supplementary loss functions, training schedules, and speaker and language diversity for Dravidian and Indo-Aryan languages. Based on this, we identify monolingual models with FastPitch and HiFi-GAN V1, trained jointly on male and female speakers to perform the best. With this setup, we train and evaluate TTS models for 13 languages and find our models to significantly improve upon existing models in all languages as measured by mean opinion scores. We open-source all models on the [Bhashini platform](https://bhashini.gov.in/ulca/model/explore-models).

**TL;DR:** We open-source SOTA Text-To-Speech models for 13 Indian languages: *Assamese, Bengali, Bodo, Gujarati, Hindi, Kannada, Malayalam, Manipuri, Marathi, Odia, Rajasthani, Tamil and Telugu*.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-assamese-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-assamese-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-bengali-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-bengali-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-bodo-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-bodo-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-gujarati-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-gujarati-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-hindi-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-hindi-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-kannada-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-kannada-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-malayalam-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-malayalam-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-manipuri-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-manipuri-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-marathi-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-marathi-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-rajasthani-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-rajasthani-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-tamil-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-tamil-on-indictts?p=towards-building-text-to-speech-systems-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/towards-building-text-to-speech-systems-for/speech-synthesis-telugu-on-indictts)](https://paperswithcode.com/sota/speech-synthesis-telugu-on-indictts?p=towards-building-text-to-speech-systems-for)
	

**Authors:** Gokul Karthik Kumar*, Praveen S V*, Pratyush Kumar, Mitesh M. Khapra, Karthik Nandakumar

**[[ArXiv Preprint](https://arxiv.org/abs/2211.09536)] [[Audio Samples](https://models.ai4bharat.org/#/tts/samples)] [[Try It Live](https://models.ai4bharat.org/#/tts)] [[Video](https://youtu.be/I3eo8IUAP7s)]**

## Unified architecture of our TTS system
<img src='images/architecture.png' width=1024>

## Results
<img src='images/evaluation.png' width=1024>

## Setup:
### Environment Setup:
```
# 1. Create environment
sudo apt-get install libsndfile1-dev ffmpeg enchant
conda create -n tts-env
conda activate tts-env

# 2. Setup PyTorch
pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Setup Trainer
git clone https://github.com/gokulkarthik/Trainer 

cd Trainer
pip3 install -e .[all]
cd ..
[or]
cp Trainer/trainer/logging/wandb_logger.py to the local Trainer installation # fixed wandb logger
cp Trainer/trainer/trainer.py to the local Trainer installation # fixed model.module.test_log and added code to log epoch 
add `gpus = [str(gpu) for gpu in gpus]` in line 53 of trainer/distribute.py

# 4. Setup TTS
git clone https://github.com/gokulkarthik/TTS 

cd TTS
pip3 install -e .[all]
cd ..
[or]
cp TTS/TTS/bin/synthesize.py to the local TTS installation # added multiple output support for TTS.bin.synthesis

# 5. Install other requirements
> pip3 install -r requirements.txt
```


### Data Setup:
1. Format IndicTTS dataset in LJSpeech format using [preprocessing/FormatDatasets.ipynb](./preprocessing/FormatDatasets.ipynb)
2. Analyze IndicTTS dataset to check TTS suitability using [preprocessing/AnalyzeDataset.ipynb](./preprocessing/AnalyzeDataset.ipynb)

### Training Steps:
1. Set the configuration with [main.py](./main.py), [vocoder.py](./vocoder.py), [configs](./configs) and [run.sh](./run.sh). Make sure to update the CUDA_VISIBLE_DEVICES in all these files.
2. Train and test by executing `sh run.sh`

### Inference:
Trained model weight and config files can be downloaded at [this link.](https://github.com/AI4Bharat/Indic-TTS/releases/tag/v1-checkpoints-release)

```
python3 -m TTS.bin.synthesize --text <TEXT> \
    --model_path <LANG>/fastpitch/best_model.pth \
    --config_path <LANG>/config.json \
    --vocoder_path <LANG>/hifigan/best_model.pth \
    --vocoder_config_path <LANG>/hifigan/config.json \
    --out_path <OUT_PATH>
```

---
Code Reference: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)
`
