

python generate_ctc_cmlm.py data-bin/wmt16.en-ro \
    --path output/checkpoint20.pt \
    --task translation_self \
    --remove-bpe \
    --max-sentences 2 \
    --decoding-iterations 10 \
    --decoding-strategy ctc_mask_predict