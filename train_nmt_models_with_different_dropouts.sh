#!/bin/bash

# just start by taking the same preprocessed data as for the baseline model

mkdir dropout
mkdir dropout/checkpoints
mkdir dropout/checkpoints/dropout_0-4
mkdir dropout/checkpoints/dropout_0-6

# train model with dropout rate of 0.4
python3 train.py --save-dir dropout/checkpoints/dropout_0-4 --dropout 0.4 --cuda True

# train model with dropout rate of 0.6
python3 train.py --save-dir dropout/checkpoints/dropout_0-6 --dropout 0.6 --cuda True

# translate test set with model trained with dropout rate of 0.4
mkdir dropout/translations
python3 translate.py --checkpoint-path dropout/checkpoints/dropout_0-4/checkpoint_best.pt --output dropout/translations/translation_0-4.txt --cuda True

# translate test set with model trained with dropout rate of 0.6
python3 translate.py --checkpoint-path dropout/checkpoints/dropout_0-6/checkpoint_best.pt --output dropout/translations/translation_0-6.txt --cuda True

# post-process translations and calculate BLEU score
sh postprocess.sh dropout/translations/translation_0-4.txt dropout/translations/translation_0-4_post.txt en
cat dropout/translations/translation_0-4_post.txt | sacrebleu baseline/raw_data/test.en

sh postprocess.sh dropout/translations/translation_0-6.txt dropout/translations/translation_0-6_post.txt en
cat dropout/translations/translation_0-6_post.txt | sacrebleu baseline/raw_data/test.en
