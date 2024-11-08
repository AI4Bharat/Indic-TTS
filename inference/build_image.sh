#!/bin/bash

# Indo Aryan TTS
# bash build_image.sh as bn gu hi mr or pa raj

# Dravidian TTS
# bash build_image.sh kn ml ta te

# Misc TTS
# bash build_image.sh brx en+hi mni

# Array of URLs to download with language codes as keys
declare -A urls=(
    ["as"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/as.zip"
    ["bn"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/bn.zip"
    ["brx"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/brx.zip"
    ["en+hi"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/en+hi.zip"
    ["gu"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/gu.zip"
    ["hi"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/hi.zip"
    ["kn"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/kn.zip"
    ["ml"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/ml.zip"
    ["mni"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/mni.zip"
    ["mr"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/mr.zip"
    ["or"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/or.zip"
    ["pa"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/pa.zip"
    ["ta"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/ta.zip"
    ["te"]="https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/te.zip"
)

# Output directory
output_dir="checkpoints"
mkdir -p "$output_dir"

# Check if specific language codes are provided as command-line arguments
if [ "$#" -eq 0 ]; then
    # If no arguments are provided, download all checkpoints
    languages=("${!urls[@]}")
else
    # Otherwise, download only the specified checkpoints
    languages=("$@")
fi

# Loop through each specified language
for lang in "${languages[@]}"; do
    url="${urls[$lang]}"
    if [ -z "$url" ]; then
        echo "No URL found for language code: $lang. Skipping."
        continue
    fi

    # Download the ZIP file
    echo "Downloading $url..."
    curl -L "$url" -o temp.zip

    # Unzip the downloaded file
    echo "Unzipping temp.zip..."
    unzip temp.zip -d "$output_dir"

    # Remove the ZIP file
    echo "Cleaning up temp.zip..."
    rm temp.zip

    echo "Done with $url."
done

docker build -t ai4bharat/triton-indic-tts .