#!/bin/bash

# use the best results of experiments with different BPE vocabulary sizes and dropout rates
# train NMT model with BPE splits (vocabulary of 6k subwords) and dropout rate 0.4
# (this assumes that bpe model has already been applied to training data)
mkdir combined_bpe_dropout
mkdir combined_bpe_dropout/checkpoints

python3 train.py --data bpe/prepared_data/6k --save-dir combined_bpe_dropout/checkpoints --dropout 0.4 --bpe_vocab_size _6k --cuda True

mkdir combined_bpe_dropout/translations
python3 translate.py --checkpoint-path combined_bpe_dropout/checkpoints/checkpoint_best.pt --data bpe/prepared_data/6k  --bpe_vocab_size _6k --output combined_bpe_dropout/translations/translation.txt --cuda True

cat combined_bpe_dropout/translations/translation.txt | sed 's/\@\@ //g' > combined_bpe_dropout/translations/translation_no_bpe.txt
sh postprocess.sh combined_bpe_dropout/translations/translation_no_bpe.txt combined_bpe_dropout/translations/translation_post.txt en
cat combined_bpe_dropout/translations/translation_post.txt | sacrebleu baseline/raw_data/test.en
