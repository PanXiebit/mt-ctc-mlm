model_dir=output
src=en
tgt=ro

python train.py data-bin/wmt16.${src}-${tgt} \
    --arch bert_transformer_seq2seq_ctc \
    --share-all-embeddings \
    --criterion label_smoothed_cross_entropy_ctc \
    --label-smoothing 0.1 \
    --lr 5e-4 \
    --warmup-init-lr 1e-7 \
    --min-lr 1e-9 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --optimizer adam \
    --adam-betas '(0.9, 0.999)' \
    --adam-eps 1e-6 \
    --task translation_self \
    --max-tokens 300 \
    --weight-decay 0.01 \
    --dropout 0.3 \
    --encoder-layers 6 \
    --ctc-encoder-layers 1 \
    --encoder-embed-dim 512 \
    --decoder-layers 6 \
    --decoder-embed-dim 512 \
    --max-source-positions 128 \
    --max-target-positions 128 \
    --mlm-weights 5.0 \
    --ctc-weights 1.0 \
    --max-update 300000 \
    --seed 0 \
    --save-dir ${model_dir}