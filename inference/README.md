# AI4Bharat-TTS Inference

Text-to-Speech models trained by [AI4Bhārat](https://ai4bharat.iitm.ac.in) for 15 major languages spoken in the Indian Republic, supporting both male and female speakers.

## Details

### Dataset

The models were trained using [the TTS dataset built by IIT-M's SMT Lab](https://www.iitm.ac.in/donlab/tts/database.php).

### Languages

The list of 15 languages include Indian-English, 2 Tibeto-Burman languages (Bodo & Meitei from northeast) and 12 Indic languages (4 Dravidian from South-India and 8 Indo-Aryan from Northern-India).

| **Language** | **Code** | **Speakers** | **Script**             | **Family**    | **Native Region**         |
|--------------|----------|--------------|------------------------|---------------|---------------------------|
| Assamese     | as       | male, female | Eastern-Nagari         | Indo-Aryan    | Assam                     |
| Bangla       | bn       | male, female | Eastern-Nagari         | Indo-Aryan    | West-Bengal, Bangladesh   |
| Boro         | brx      |       female | DevaNagari             | Tibeto-Burman | Bodoland Territory        |
| English      | en       | male, female | Roman                  | European      | -- Lingua franca --       |
| Hinglish     | en+hi    | male, female | Code-mixed             | Indo-European |                           |
| Gujarati     | gu       | male, female | Gujrati                | Indo-Aryan    | Gujarat                   |
| Hindi        | hi       | male, female | DevaNagari             | Indo-Aryan    | Hindi Belt                |
| Kannada      | kn       | male, female | Kannada                | Dravidian     | Karnataka                 |
| Malayalam    | ml       | male, female | Malayalam              | Dravidian     | Kerala                    |
| Manipuri     | mni      | male, female | Meetei, Eastern-Nagari | Tibeto-Burman | Manipur                   |
| Marathi      | mr       | male, female | DevaNagari             | Indo-Aryan    | Maharashtra               |
| Oriya        | or       | male, female | Odia                   | Indo-Aryan    | Odisha                    |
| Panjabi      | pa       | male, female | Gurumukhi              | Indo-Aryan    | Eastern-Punjab            |
| Rajasthani   | raj      | male, female | DevaNagari             | Indo-Aryan    | Rajasthan                 |
| Tamil        | ta       | male, female | Tamil                  | Dravidian     | Tamil Nadu                |
| Telugu       | te       | male, female | Telugu                 | Dravidian     | Andhra Pradesh, Telangana |

## Usage

### Pre-requisites

1. Python 3.9+
2. If Linux, install the following dependencies:
```
cd inference
sudo apt-get install libsndfile1-dev ffmpeg enchant
```
(For any other OS, we wish you best of luck)
3. `pip install -r requirements-ml.txt requirements-utils.txt`
4. [Download the models from here](https://github.com/AI4Bharat/Indic-TTS/releases), place them inside a new folder named `checkpoints` and unzip them.

### Running server

```
pip install -r requirements-server.txt
uvicorn server:api
```
