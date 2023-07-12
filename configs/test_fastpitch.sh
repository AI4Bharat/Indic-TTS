python3 -m TTS.bin.synthesize --text "../../datasets/indictts/ta/samples.csv" \
    --model_path output/store/ta/fastpitch/best_model.pth \
    --config_path output/store/ta/fastpitch/config.json \
    --vocoder_path output_vocoder/store/ta/hifigan/best_model.pth \
    --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
    --out_path output_wavs/samples

python3 scripts/evaluate_mcd.py \
    output_wavs/samples/ \
    data_dir/indictts/ta/wavs-22k

python3 scripts/evaluate_f0.py \
    output_wavs/samples/ \
    /data_dir/indictts/ta/wavs-22k
