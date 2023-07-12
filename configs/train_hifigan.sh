CUDA_VISIBLE_DEVICES='0' python3 vocoder.py --dataset_name indictts \
    --language mr \
    --speaker male \
    --batch_size 32 \
    --batch_size_eval 32 \
    --epochs 5000 \
    --port 10004 \
    --mixed_precision t 