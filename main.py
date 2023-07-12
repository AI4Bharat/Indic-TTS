import argparse
import os
import string

import numpy as np
import pandas as pd
import torch

from argparse import Namespace
from torch.utils.data import DataLoader
from trainer import Trainer, TrainerArgs
from TTS.config import load_config
from TTS.tts.configs.align_tts_config import AlignTTSConfig
from TTS.tts.configs.fast_pitch_config import FastPitchConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseAudioConfig, BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import TTSDataset, load_tts_samples
from TTS.tts.models import setup_model
from TTS.tts.models.align_tts import AlignTTS
from TTS.tts.models.forward_tts import ForwardTTS, ForwardTTSArgs
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_checkpoint
from tqdm.auto import tqdm

from utils import str2bool


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Traning and evaluation script for acoustic / e2e TTS model ')

    # dataset parameters
    parser.add_argument('--dataset_name', default='indictts', choices=['ljspeech', 'indictts', 'googletts'])
    parser.add_argument('--language', default='ta', choices=['en', 'ta', 'te', 'kn', 'ml', 'hi', 'mr', 'bn', 'gu', 'or', 'as', 'raj', 'mni', 'brx', 'all'])
    parser.add_argument('--dataset_path', default='/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/{}/{}', type=str) # dataset_name, language #CHANGE
    parser.add_argument('--speaker', default='all') # eg. all, male, female, ...
    parser.add_argument('--use_phonemes', default=False, type=str2bool)
    parser.add_argument('--phoneme_language', default='en-us', choices=['en-us'])
    parser.add_argument('--add_blank', default=False, type=str2bool)
    parser.add_argument('--text_cleaner', default='multilingual_cleaners', choices=['multilingual_cleaners'])
    parser.add_argument('--eval_split_size', default=0.01)
    parser.add_argument('--min_audio_len', default=1)
    parser.add_argument('--max_audio_len', default=float("inf")) # 20*22050
    parser.add_argument('--min_text_len', default=1)
    parser.add_argument('--max_text_len', default=float("inf")) # 400
    parser.add_argument('--audio_config', default='without_norm', choices=['without_norm', 'with_norm'])

    # model parameters
    parser.add_argument('--model', default='glowtts', choices=['glowtts', 'vits', 'fastpitch', 'tacotron2', 'aligntts'])
    parser.add_argument('--hidden_channels', default=512, type=int)
    parser.add_argument('--use_speaker_embedding', default=True, type=str2bool)
    parser.add_argument('--use_d_vector_file', default=False, type=str2bool)
    parser.add_argument('--d_vector_file', default="", type=str)
    parser.add_argument('--d_vector_dim', default=512, type=int)
    parser.add_argument('--speaker_encoder_model_path', default='', type=str) 
    parser.add_argument('--speaker_encoder_config_path', default='', type=str) 
    parser.add_argument('--use_speaker_encoder_as_loss', default=False, type=str2bool) # only supported in vits, fastpitch
    parser.add_argument('--use_ssim_loss', default=False, type=str2bool) # only supported in fastpitch
    parser.add_argument('--vocoder_path', default=None, type=str) # external vocoder for speaker encoder loss in fastpitch
    parser.add_argument('--vocoder_config_path', default=None, type=str)  # external vocoder for speaker encoder loss in fastpitch
    parser.add_argument('--use_style_encoder', default=False, type=str2bool)
    parser.add_argument('--use_aligner', default=True, type=str2bool) # for fastspeech, fastpitch
    parser.add_argument('--use_separate_optimizers', default=False, type=str2bool) # for aligner in fastspeech, fastpitch
    parser.add_argument('--use_pre_computed_alignments', default=False, type=str2bool) # for fastspeech, fastpitch
    parser.add_argument('--pretrained_checkpoint_path', default=None, type=str) # to load pretrained weights
    parser.add_argument('--attention_mask_model_path', default='output/store/ta/fastpitch/best_model.pth', type=str) # set if use_aligner==False and use_pre_computed_alignments==False #CHANGE
    parser.add_argument('--attention_mask_config_path', default='output/store/ta/fastpitch/config.json', type=str) # set if use_aligner==False and use_pre_computed_alignments==False #CHANGE
    parser.add_argument('--attention_mask_meta_file_name', default='meta_file_attn_mask.txt', type=str) # dataset_name, language # set if use_aligner==False #CHANGE

    # training parameters
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--aligner_epochs', default=1000, type=int) # For FastPitch
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--batch_size_eval', default=8, type=int)
    parser.add_argument('--batch_group_size', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_workers_eval', default=8, type=int)
    parser.add_argument('--mixed_precision', default=False, type=str2bool)
    parser.add_argument('--compute_input_seq_cache', default=False, type=str2bool)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_scheduler', default='NoamLR', choices=['NoamLR', 'StepLR', 'LinearLR', 'CyclicLR', 'NoamLRStepConstant', 'NoamLRStepDecay'])
    parser.add_argument('--lr_scheduler_warmup_steps', default=4000, type=int) # NoamLR
    parser.add_argument('--lr_scheduler_step_size', default=500, type=int) # StepLR
    parser.add_argument('--lr_scheduler_threshold_step', default=500, type=int) # NoamLRStep+
    parser.add_argument('--lr_scheduler_aligner', default='NoamLR', choices=['NoamLR', 'StepLR', 'LinearLR', 'CyclicLR', 'NoamLRStepConstant', 'NoamLRStepDecay'])
    parser.add_argument('--lr_scheduler_gamma', default=0.1, type=float) # StepLR, LinearLR, CyclicLR

    # training - logging parameters 
    parser.add_argument('--run_description', default='None', type=str)
    parser.add_argument('--output_path', default='output', type=str)
    parser.add_argument('--test_delay_epochs', default=0, type=int)   
    parser.add_argument('--print_step', default=100, type=int)
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--save_step', default=10000, type=int)
    parser.add_argument('--save_n_checkpoints', default=1, type=int)
    parser.add_argument('--save_best_after', default=10000, type=int)
    parser.add_argument('--target_loss', default=None)
    parser.add_argument('--print_eval', default=False, type=str2bool)
    parser.add_argument('--run_eval', default=True, type=str2bool)
    
    # distributed training parameters
    parser.add_argument('--port', default=54321, type=int)
    parser.add_argument('--continue_path', default="", type=str)
    parser.add_argument('--restore_path', default="", type=str)
    parser.add_argument('--group_id', default="", type=str)
    parser.add_argument('--use_ddp', default=True, type=bool)
    parser.add_argument('--rank', default=0, type=int)
    #parser.add_argument('--gpus', default='0', type=str)

    # vits
    parser.add_argument('--use_sdp', default=True, type=str2bool)

    return parser


def formatter_indictts(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs-22k", cols[0] + ".wav")
            text = cols[1].strip()
            speaker_name = cols[2].strip()
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name})
    return items


def filter_speaker(samples, speaker):
    if speaker == 'all':
        return samples
    samples = [sample for sample in samples if sample['speaker_name']==speaker]
    return samples


def get_lang_chars(language):
    if language == 'ta':
        lang_chars_df = pd.read_csv('chars/Characters-Tamil.csv')
        lang_chars = sorted(list(set(list("".join(lang_chars_df['Character'].values.tolist())))))
        print(lang_chars, len(lang_chars))
        print("".join(lang_chars))
        lang_chars_extra = ['ௗ', 'ஹ', 'ஜ', 'ஸ', 'ஷ']
        lang_chars_extra = sorted(list(set(list("".join(lang_chars_extra)))))
        print(lang_chars_extra, len(lang_chars_extra))
        print("".join(lang_chars_extra))
        lang_chars = lang_chars + lang_chars_extra

    elif language == 'hi':
        lang_chars_df = pd.read_csv('chars/Characters-Hindi.csv')
        lang_chars = sorted(list(set(list("".join(lang_chars_df['Character'].values.tolist())))))
        print(lang_chars, len(lang_chars))
        print("".join(lang_chars))
        lang_chars_extra = []
        lang_chars_extra = sorted(list(set(list("".join(lang_chars_extra)))))
        print(lang_chars_extra, len(lang_chars_extra))
        print("".join(lang_chars_extra))
        lang_chars = lang_chars + lang_chars_extra

    elif language == 'en':
        lang_chars = string.ascii_lowercase

    return lang_chars


def get_test_sentences(language):
    if language == 'ta':
        test_sentences = [
                "நேஷனல் ஹெரால்ட் ஊழல் குற்றச்சாட்டு தொடர்பாக, காங்கிரஸ் நாடாளுமன்ற உறுப்பினர் ராகுல் காந்தியிடம், அமலாக்கத்துறை, திங்கள் கிழமையன்று பத்து மணி நேரத்திற்கும் மேலாக விசாரணை நடத்திய நிலையில், செவ்வாய்க்கிழமை மீண்டும் விசாரணைக்கு ஆஜராகிறார்.",
                "ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின் பேரிலும் நுட்பமாகவும் நடத்துகிறார்.",
            ]

    elif language == 'en':
        test_sentences = [
                "Brazilian police say a suspect has confessed to burying the bodies of missing British journalist Dom Phillips and indigenous expert Bruno Pereira.",
                "Protests have erupted in India over a new reform scheme to hire soldiers for a fixed term for the armed forces",
            ]
        
    elif language == 'mr':
        test_sentences = [
                "मविआ सरकार अल्पमतात आल्यानंतर अनेक निर्णय घेतले: मुख्यमंत्री एकनाथ शिंदे यांचा आरोप.",
                "वर्ध्यात भदाडी नदीच्या पुलावर कार डिव्हायडरला धडकून भीषण अपघात, दोघे गंभीर जखमी.",
            ]

    elif language == 'as':
        test_sentences = [
                "দেউতাই উইলত স্পষ্টকৈ সেইখিনি মোৰ নামত লিখি দি গৈছে",
                "গতিকে শিক্ষাৰ বাবেও এনে এক পূৰ্ব প্ৰস্তুত পৰি‌ৱেশ এটাত",
            ]

    elif language == 'bn':
        test_sentences = [
                "লোডশেডিংয়ের কল্যাণে পুজোর দুসপ্তাহ আগে কেনাকাটার মাহেন্দ্রক্ষণে, দোকানে শোভা পাচ্ছে, মোমবাতি",
                "এক চন্দরা নির্দোষ হইয়াও, আইনের আপাত নিশ্ছিদ্র জালে পড়িয়া প্রাণ দিয়াছিল",
            ]

    elif language == 'brx':
        test_sentences = [
                "गावनि गोजाम गामि नवथिखौ हरखाब नागारनानै गोदान हादानाव गावखौ दिदोमै फसंथा फित्राय हाबाया जोबोद गोब्राब जायोलै गोमजोर",
                "सानहाबदों आं मोथे मोथो",
            ]

    elif language == 'gu':
        test_sentences = [
                "ઓગણીસો છત્રીસ માં, પ્રથમવાર, એક્રેલીક સેફટી ગ્લાસનું, ઉત્પાદન, શરુ થઈ ગયું.",
                "વ્યાયામ પછી પ્રોટીન લેવાથી, સ્નાયુની જે પેશીયોને હાનિ પ્હોંચી હોય છે.",
            ]

    elif language == 'hi':
        test_sentences = [
                "बिहार, राजस्थान और उत्तर प्रदेश से लेकर हरियाणा, मध्य प्रदेश एवं उत्तराखंड में सेना में भर्ती से जुड़ी 'अग्निपथ स्कीम' का विरोध जारी है.",
                "संयुक्त अरब अमीरात यानी यूएई ने बुधवार को एक फ़ैसला लिया कि अगले चार महीनों तक वो भारत से ख़रीदा हुआ गेहूँ को किसी और को नहीं बेचेगा.",
            ]

    elif language == 'kn':
        test_sentences = [
                "ಯಾವುದು ನಿಜ ಯಾವುದು ಸುಳ್ಳು ಎನ್ನುವ ಬಗ್ಗೆ ಚಿಂತಿಸಿ.",
                "ಶಕ್ತಿ ಇದ್ದರೆನ್ನೊಡನೆ ಜಗಳಕ್ಕೆ ಬಾ",
            ]


    elif language == 'ml':
        test_sentences = [
                "ശിലായുഗകാലം മുതൽ മനുഷ്യർ ജ്യാമിതീയ രൂപങ്ങൾ ഉപയോഗിച്ചുവരുന്നു",
                "വാഹനാപകടത്തിൽ പരുക്കേറ്റ അധ്യാപിക മരിച്ചു",
            ]

    elif language == 'mni':
        test_sentences = [
                "মথং মথং, অসুম কাখিবনা.",
                "থেবনা ঙাশিংদু অমমম্তা ইল্লে.",
            ]

    elif language == 'mr':
        test_sentences = [
                "म्हणुनच महाराच बिरुद मी मानान वागवल",
                "घोडयावरून खाली उतरताना घोडेस्वार वृध्दाला म्हणाला, बाबा एवढया कडाक्याच्या थंडीत नदी कडेला तुम्ही किती वेळ बसला होतात.",
            ]

    elif language == 'or':
        test_sentences = [
                "ସାମାନ୍ୟ ଗୋଟିଏ ବାଳକ, ସେ କ’ଣ ମହାଭାରତ ଯୁଦ୍ଧରେ ଲଢ଼ିବ ",
                "ଏ ଘଟଣା ଦେଖିବାକୁ ଶହ ଶହ ଲୋକ ଧାଇଁଲେ ",
            ]

    elif language == 'raj':
        test_sentences = [
                "कन्हैयालाल सेठिया इत्याद अनुपम काव्य कृतियां है, इंया ई, प्रकति काव्य री दीठ सूं, बादळी, लू",
                "नई बीनणियां रो घूंघटो नाक रे ऊपर ऊपर पड़यो सावे है",
            ]

    elif language == 'te':
        test_sentences = [
                "సింహం అడ్డువచ్చి, తప్పుకో శిక్ష విధించవలసింది నేను అని కోతిని అఙ్ఞాపించింది నక్కకేసి తిరిగి మంత్రి పుంగవా ఈ మూషికాధముడు చోరుడు అని నీకు ఎలా తెలిసింది అని అడిగింది.",
                "ఈ మాటలు వింటూనే గాలవుడు, కువలయాశ్వాన్ని ఎక్కి, శత్రుజిత్తువద్దకు వెళ్లి, ఋతుధ్వజుణ్ణి పంపమని కోరాడు, ఋతుధ్వజుడు, కువలయాశ్వాన్ని ఎక్కి, గాలవుడి వెంట, ఆయన ఆశ్రమానికి వెళ్ళాడు.",
            ]

    elif language == 'all':
        test_sentences = [
                "ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின் பேரிலும் நுட்பமாகவும் நடத்துகிறார்.",
                "ఇక బిన్ లాడెన్ తర్వాతి అగ్ర నాయకులు అయ్‌మన్ అల్ జవహరి తదితర ముఖ్యుల 'తలలు నరికి ఈటెలకు గుచ్చండి' అనేవి ఇతర ఆదేశాలు.",
                "ಕೆಲ ದಿನಗಳಿಂದ ಮಳೆ ಕಡಿಮೆಯಾದಂತೆ ತೋರಿದ್ದರೂ ಕಳೆದ ಎರಡು ದಿನಗಳಲ್ಲಿ ರಾಜ್ಯದ ಹಲವೆಡೆ ಮತ್ತೆ ಮಳೆ ಸುರಿದಿದ್ದು ಇದರ ಪರಿಣಾಮದಿಂದಾಗಿ ಮತ್ತೆ ನೀರಿನ ಹರಿವು ಏರುವ ಪಥದಲ್ಲಿದೆ.",
                "കോമണ്‍വെല്‍ത്ത് ഗെയിംസ് വനിതാ ക്രിക്കറ്റ് സെമി ഫൈനലില്‍ ഇംഗ്ലണ്ടിനെ ആവേശപ്പോരില്‍ വീഴ്ത്തി ഇന്ത്യ ഫൈനലിലെത്തി."
            ]

    else:
        raise ValueError("test_sentences are not defined")

    return test_sentences


def compute_attention_masks(model_path, config_path, meta_save_path, data_path, dataset_metafile, args, use_cuda=True):
    dataset_name = args.dataset_name
    language = args.language
    batch_size = 16
    meta_save_path = meta_save_path.format(dataset_name, language)

    C = load_config(config_path)
    ap = AudioProcessor(**C.audio)

    # load the model
    model = setup_model(C)
    model, _ = load_checkpoint(model, model_path, use_cuda, True)

    # data loader
    dataset_config = BaseDatasetConfig(
        name=dataset_name, 
        meta_file_train=dataset_metafile, 
        path=data_path, 
        language=language
    )
    samples, _ = load_tts_samples(
        dataset_config, 
        eval_split=False,
        formatter=formatter_indictts
    )

    dataset = TTSDataset(
        outputs_per_step=model.decoder.r if "r" in vars(model.decoder) else 1,
        compute_linear_spec=False,
        ap=ap,
        samples=samples,
        tokenizer=model.tokenizer,
        phoneme_cache_path=C.phoneme_cache_path,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        drop_last=False,
    )

    # compute attentions
    file_paths = []
    with torch.no_grad():
        for data in tqdm(loader):
            # setup input data
            text_input = data["token_id"]
            text_lengths = data["token_id_lengths"]
            #linear_input = data[3]
            mel_input = data["mel"]
            mel_lengths = data["mel_lengths"]
            #stop_targets = data[6]
            item_idxs = data["item_idxs"]

            # dispatch data to GPU
            if use_cuda:
                text_input = text_input.cuda()
                text_lengths = text_lengths.cuda()
                mel_input = mel_input.cuda()
                mel_lengths = mel_lengths.cuda()

            if C.model == 'glowtts':
                model_outputs = model.forward(text_input, text_lengths, mel_input, mel_lengths)
                #model_outputs = model.inference(text_input, text_lengths, mel_input, mel_lengths)
            elif C.model == 'fast_pitch':
                model_outputs = model.inference2(text_input, text_lengths)
            else:
                raise ValueError

            alignments = model_outputs["alignments"].detach()
            for idx, alignment in enumerate(alignments):
                item_idx = item_idxs[idx]
                # interpolate if r > 1
                alignment = (
                    torch.nn.functional.interpolate(
                        alignment.transpose(0, 1).unsqueeze(0),
                        size=None,
                        scale_factor=model.decoder.r if "r" in vars(model.decoder) else 1,
                        mode="nearest",
                        align_corners=None,
                        recompute_scale_factor=None,
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
                # remove paddings
                alignment = alignment[: mel_lengths[idx], : text_lengths[idx]].cpu().numpy()
                # set file paths
                wav_file_name = os.path.basename(item_idx)
                align_file_name = os.path.splitext(wav_file_name)[0] + "_attn.npy"
                file_path = item_idx.replace(wav_file_name, align_file_name)
                # save output
                wav_file_abs_path = os.path.abspath(item_idx)
                file_abs_path = os.path.abspath(file_path)
                file_paths.append([wav_file_abs_path, file_abs_path])
                np.save(file_path, alignment)

        # output metafile
        with open(meta_save_path, "w", encoding="utf-8") as f:
            for p in file_paths:
                f.write(f"{p[0]}|{p[1]}\n")
        print(f" >> Metafile created: {meta_save_path}")

    return True
    

def main(args):

    if args.speaker == 'all':
        meta_file_train="metadata_train.csv"
        meta_file_val="metadata_test.csv"
    else:
        meta_file_train=f"metadata_train_{args.speaker}.csv"
        meta_file_val=f"metadata_test_{args.speaker}.csv"

    # set dataset config
    dataset_config = BaseDatasetConfig(
        name=args.dataset_name, 
        meta_file_train=meta_file_train, 
        meta_file_val=meta_file_val,
        path=args.dataset_path.format(args.dataset_name, args.language), 
        language=args.language
    )

    #lang_chars = get_lang_chars(args.language)
    samples, _ = load_tts_samples(
        dataset_config, 
        eval_split=False,
        formatter=formatter_indictts)
    samples = filter_speaker(samples, args.speaker)
    texts = "".join(item["text"] for item in samples)
    lang_chars = sorted(list(set(texts)))
    print(lang_chars, len(lang_chars))
    del samples, texts

    # set audio config
    audio_config = BaseAudioConfig(
        trim_db=60.0, # default: 45
        #mel_fmin=0.0,  # default: 0
        mel_fmax=8000, # default: None
        log_func="np.log", # default: np.log10
        spec_gain=1.0, # default: 20
        signal_norm=False, # default: True
    )

    audio_configs = {
        "without_norm": BaseAudioConfig(
            trim_db=60.0, # default: 45
            #mel_fmin=0.0,  # default: 0
            mel_fmax=8000, # default: None
            log_func="np.log", # default: np.log10
            spec_gain=1.0, # default: 20
            signal_norm=False, # default: True
        ), 
        "with_norm": BaseAudioConfig(
            trim_db=60.0, # default: 45
            #mel_fmin=0.0,  # default: 0
            mel_fmax=8000, # default: None
            log_func="np.log10", # default: np.log10
            spec_gain=20, # default: 20
            signal_norm=True, # default: True
        ), 
    }
    audio_config = audio_configs[args.audio_config]

    # set characters config
    characters_config = CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        #characters="!¡'(),-.:;¿?$%&‘’‚“`”„" + "".join(lang_chars),
        characters="".join(lang_chars),
        punctuations="!¡'(),-.:;¿? ",
        phonemes=None
    )

    if args.lr_scheduler == 'NoamLR':
        lr_scheduler_params = {
            "warmup_steps": args.lr_scheduler_warmup_steps
        }
    elif args.lr_scheduler == 'StepLR':
        lr_scheduler_params = {
            "step_size": args.lr_scheduler_step_size,
            "gamma": args.lr_scheduler_gamma
        }
    elif args.lr_scheduler == 'LinearLR':
        lr_scheduler_params = {
            "start_factor": args.lr_scheduler_gamma,
            "total_iters": args.lr_scheduler_warmup_steps
        }
    elif args.lr_scheduler == 'CyclicLR':
        lr_scheduler_params = {
            "base_lr": args.lr * args.lr_scheduler_gamma,
            "max_lr": args.lr,
            "cycle_momentum": False
        }
    elif args.lr_scheduler in ['NoamLRStepConstant', 'NoamLRStepDecay'] :
        lr_scheduler_params = {
            "warmup_steps": args.lr_scheduler_warmup_steps,
            "threshold_step": args.lr_scheduler_threshold_step
        }
    else:
        raise NotImplementedError()

    if args.lr_scheduler_aligner == 'NoamLR':
        lr_scheduler_aligner_params = {
            "warmup_steps": args.lr_scheduler_warmup_steps
        }
    elif args.lr_scheduler_aligner == 'StepLR':
        lr_scheduler_aligner_params = {
            "step_size": args.lr_scheduler_step_size
        }
    elif args.lr_scheduler_aligner in ['NoamLRStepConstant', 'NoamLRStepDecay'] :
        lr_scheduler_aligner_params = {
            "warmup_steps": args.lr_scheduler_warmup_steps,
            "threshold_step": args.lr_scheduler_threshold_step
        }
    else:
        raise NotImplementedError()


    # set base tts config
    base_tts_config = Namespace(
        # input representation
        audio=audio_config,
        use_phonemes=args.use_phonemes,
        phoneme_language=args.phoneme_language,
        compute_input_seq_cache=args.compute_input_seq_cache,
        text_cleaner=args.text_cleaner,
        phoneme_cache_path=os.path.join(args.output_path, "phoneme_cache"),
        characters=characters_config,
        add_blank=args.add_blank,
        # dataset
        datasets=[dataset_config],
        min_audio_len=args.min_audio_len,
        max_audio_len=args.max_audio_len,
        min_text_len=args.min_text_len,
        max_text_len=args.max_text_len,
        # data loading
        num_loader_workers=args.num_workers,
        num_eval_loader_workers=args.num_workers_eval,
        # model
        use_d_vector_file=args.use_d_vector_file,
        d_vector_file=args.d_vector_file,
        d_vector_dim=args.d_vector_dim,
        # trainer - run
        output_path=args.output_path,
        project_name='indic-tts-acoustic',
        run_name=f'{args.language}_{args.model}_{args.dataset_name}_{args.speaker}_{args.run_description}',
        run_description=args.run_description,
        # trainer - loggging
        print_step=args.print_step,
        plot_step=args.plot_step,
        dashboard_logger='wandb',
        wandb_entity='indic-asr',
        # trainer - checkpointing
        save_step=args.save_step,
        save_n_checkpoints=args.save_n_checkpoints,
        save_best_after=args.save_best_after,
        # trainer - eval
        print_eval=args.print_eval,
        run_eval=args.run_eval,
        # trainer - test
        test_delay_epochs=args.test_delay_epochs,
        # trainer - distibuted training
        distributed_url=f'tcp://localhost:{args.port}',
        # trainer - training
        mixed_precision=args.mixed_precision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size_eval,
        batch_group_size=args.batch_group_size,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_scheduler_params = lr_scheduler_params,
        # test
        #test_sentences_file=f'test_sentences/{args.language}.txt',
        test_sentences=get_test_sentences(args.language),
        eval_split_size=args.eval_split_size,
    )
    base_tts_config = vars(base_tts_config)

    # set model config 
    if args.model == 'glowtts':
        config = GlowTTSConfig(
            **base_tts_config,
            use_speaker_embedding=args.use_speaker_embedding,
        )
    elif args.model == "vits":
        vitsArgs = VitsArgs(
            use_speaker_embedding=args.use_speaker_embedding,
            use_sdp=args.use_sdp,
            use_speaker_encoder_as_loss=args.use_speaker_encoder_as_loss,
            speaker_encoder_config_path=args.speaker_encoder_config_path,
            speaker_encoder_model_path=args.speaker_encoder_model_path,
        )
        config = VitsConfig(
            **base_tts_config,
            model_args=vitsArgs,
            use_speaker_embedding=args.use_speaker_embedding,   
        )
    elif args.model == "fastpitch":

        if args.use_speaker_encoder_as_loss:
            return_wav = True
            compute_linear_spec = True
            assert args.vocoder_path is not None
            assert args.vocoder_config_path is not None
        else:
            return_wav = False
            compute_linear_spec = False
            args.vocoder_path = None
            args.vocoder_config_path = None

        config = FastPitchConfig(
            **base_tts_config,
            model_args = ForwardTTSArgs(
                use_aligner=args.use_aligner, 
                use_separate_optimizers=args.use_separate_optimizers,
                hidden_channels=args.hidden_channels,
                use_speaker_encoder_as_loss=args.use_speaker_encoder_as_loss,
                speaker_encoder_config_path=args.speaker_encoder_config_path,
                speaker_encoder_model_path=args.speaker_encoder_model_path,
                vocoder_path=args.vocoder_path,
                vocoder_config_path=args.vocoder_config_path
            ),
            use_speaker_embedding=args.use_speaker_embedding,
            use_ssim_loss = args.use_ssim_loss,
            compute_f0=True,
            f0_cache_path=os.path.join(args.output_path, "f0_cache"),
            sort_by_audio_len=True,
            max_seq_len=500000,
            return_wav= return_wav,
            compute_linear_spec=compute_linear_spec,
            aligner_epochs=args.aligner_epochs,
            lr_scheduler_aligner=args.lr_scheduler_aligner,
            lr_scheduler_aligner_params = lr_scheduler_aligner_params
        )

        if not config.model_args.use_aligner:
            metafile = 'metadata.csv'
            attention_mask_meta_save_path = f'{args.dataset_path}/{args.attention_mask_meta_file_name}'
            if not args.use_pre_computed_alignments:
                print("[START] Computing attention masks...")
                compute_attention_masks(args.attention_mask_model_path, args.attention_mask_config_path, attention_mask_meta_save_path, args.dataset_path, metafile, args)
                print("[END] Computing attention masks")
            dataset_config.meta_file_attn_mask = attention_mask_meta_save_path
        
    elif args.model == "tacotron2":
        config = Tacotron2Config(
            **base_tts_config,
            use_speaker_embedding=args.use_speaker_embedding,
            ga_alpha=0.0,
            decoder_loss_alpha=0.25,
            postnet_loss_alpha=0.25,
            postnet_diff_spec_alpha=0,
            decoder_diff_spec_alpha=0,
            decoder_ssim_alpha=0,
            postnet_ssim_alpha=0,
            r=2,
            attention_type="dynamic_convolution",
            double_decoder_consistency=False,
        )
    elif args.model == "aligntts":
        config = AlignTTSConfig(
            **base_tts_config,
        )

    # set preprocessors
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # load data
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        #eval_split_size=config.eval_split_size,
        formatter=formatter_indictts
    )
    train_samples = filter_speaker(train_samples, args.speaker)
    eval_samples = filter_speaker(eval_samples, args.speaker)
    print("Train Samples: ", len(train_samples))
    print("Eval Samples: ", len(eval_samples))
    
    # set speaker manager
    if args.use_speaker_embedding:
        speaker_manager = SpeakerManager()
        speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    elif args.use_d_vector_file:
        speaker_manager = SpeakerManager(
            d_vectors_file_path=args.d_vector_file, 
            encoder_model_path=args.speaker_encoder_model_path,
            encoder_config_path=args.speaker_encoder_config_path,
            use_cuda=True)
    else:
        speaker_manager = None
    
   
    # load model
    if args.model == 'glowtts':
        model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)
    elif args.model == 'vits':
        model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)
    elif args.model == 'fastpitch':
        model = ForwardTTS(config, ap, tokenizer, speaker_manager=speaker_manager)
    elif args.model == 'tacotron2':
        model = Tacotron2(config, ap, tokenizer, speaker_manager=speaker_manager)
    elif args.model == 'aligntts':
        model = AlignTTS(config, ap, tokenizer, speaker_manager=speaker_manager)
    if args.speaker == 'all':
        config.num_speakers = speaker_manager.num_speakers
        if hasattr(config, 'model_args') and hasattr(config.model_args, 'num_speakers'):
            config.model_args.num_speakers = speaker_manager.num_speakers
    else:
        config.num_speakers = 1
    if args.pretrained_checkpoint_path:
        checkpoint_state = torch.load(args.pretrained_checkpoint_path)['model']
        print(" > Partial model initialization...")
        model_dict = model.state_dict()
        for k, v in checkpoint_state.items():
            if k not in model_dict:
                print(" | > Layer missing in the model definition: {}".format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in checkpoint_state.items() if k in model_dict}
        # 2. filter out different size layers
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if v.numel() == model_dict[k].numel()}
        # 3. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(" | > {} / {} layers are restored.".format(len(pretrained_dict), len(model_dict)))
        missed_keys = set(model_dict.keys())-set(pretrained_dict.keys())
        print(" | > Missed Keys:", missed_keys)

    # set trainer
    trainer = Trainer(
        TrainerArgs(continue_path=args.continue_path, restore_path=args.restore_path, use_ddp=args.use_ddp, rank=args.rank, group_id=args.group_id), 
        config, 
        args.output_path, 
        model=model, 
        train_samples=train_samples, 
        eval_samples=eval_samples
    )

    # run training
    trainer.fit()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = get_arg_parser()
    args = parser.parse_args()

    args.dataset_path = args.dataset_path.format(args.dataset_name ,args.language)

    if args.use_style_encoder:
        assert args.use_speaker_embedding

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
