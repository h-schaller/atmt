#!/bin/bash

sh ./train_and_apply_bpe.sh

# preprocess for BPE vocab size 6000
python3 preprocess.py --target-lang en --source-lang de --dest-dir bpe/prepared_data/6k --train-prefix bpe/preprocessed_data/train_6k --valid-prefix bpe/preprocessed_data/valid_6k --test-prefix bpe/preprocessed_data/test_6k --tiny-train-prefix bpe/preprocessed_data/tiny_train_6k --vocab-src bpe/prepared_data/dict_6k.de --vocab-trg bpe/prepared_data/dict_6k.en

# preprocess for BPE vocab size 4000
python3 preprocess.py --target-lang en --source-lang de --dest-dir bpe/prepared_data/4k --train-prefix bpe/preprocessed_data/train_4k --valid-prefix bpe/preprocessed_data/valid_4k --test-prefix bpe/preprocessed_data/test_4k --tiny-train-prefix bpe/preprocessed_data/tiny_train_4k --vocab-src bpe/prepared_data/dict_4k.de --vocab-trg bpe/prepared_data/dict_4k.en

# preprocess for BPE vocab size 1500
python3 preprocess.py --target-lang en --source-lang de --dest-dir bpe/prepared_data/1-5k --train-prefix bpe/preprocessed_data/train_1-5k --valid-prefix bpe/preprocessed_data/valid_1-5k --test-prefix bpe/preprocessed_data/test_1-5k --tiny-train-prefix bpe/preprocessed_data/tiny_train_1-5k --vocab-src bpe/prepared_data/dict_1-5k.de --vocab-trg bpe/prepared_data/dict_1-5k.en

mv bpe/prepared_data/dict_1-5k.de bpe/prepared_data/1-5k
mv bpe/prepared_data/dict_1-5k.en bpe/prepared_data/1-5k
mv bpe/prepared_data/dict_4k.de bpe/prepared_data/4k
mv bpe/prepared_data/dict_4k.en bpe/prepared_data/4k
mv bpe/prepared_data/dict_6k.de bpe/prepared_data/6k
mv bpe/prepared_data/dict_6k.en bpe/prepared_data/6k


# train nmt model with BPE vocab size 6000
mkdir bpe/checkpoints
mkdir bpe/checkpoints/6k
mkdir bpe/checkpoints/4k
mkdir bpe/checkpoints/1-5k
python3 train.py --data bpe/prepared_data/6k --save-dir bpe/checkpoints/6k --bpe_vocab_size _6k --cuda True

# train nmt model with BPE vocab size 4000
python3 train.py --data bpe/prepared_data/4k --save-dir bpe/checkpoints/4k --bpe_vocab_size _4k --cuda True

# train nmt model with BPE vocab size 1500
python3 train.py --data bpe/prepared_data/1-5k --save-dir bpe/checkpoints/1-5k --bpe_vocab_size _1-5k --cuda True

# translate test set with model trained on 6k bpe vocab
mkdir bpe/translations
python3 translate.py --data bpe/prepared_data/6k --bpe_vocab_size _6k --checkpoint-path bpe/checkpoints/6k/checkpoint_best.pt --output bpe/translations/translation_6k.txt --cuda True

# translate test set with model trained on 4k bpe vocab
python3 translate.py --data bpe/prepared_data/4k --bpe_vocab_size _4k --checkpoint-path bpe/checkpoints/4k/checkpoint_best.pt --output bpe/translations/translation_4k.txt --cuda True

# translate test set with model trained on 1.5k bpe vocab
python3 translate.py --data bpe/prepared_data/1-5k --bpe_vocab_size _1-5k --checkpoint-path bpe/checkpoints/1-5k/checkpoint_best.pt --output bpe/translations/translation_1-5k.txt --cuda True

# post-process translations and calculate BLEU score
cat bpe/translations/translation_6k.txt | sed 's/\@\@ //g' > bpe/translations/translation_6k_no_bpe.txt
sh postprocess.sh bpe/translations/translation_6k_no_bpe.txt bpe/translations/translation_6k_post.txt en
cat bpe/translations/translation_6k_post.txt | sacrebleu baseline/raw_data/test.en

cat bpe/translations/translation_4k.txt | sed 's/\@\@ //g' > bpe/translations/translation_4k_no_bpe.txt
sh postprocess.sh bpe/translations/translation_4k_no_bpe.txt bpe/translations/translation_4k_post.txt en
cat bpe/translations/translation_4k_post.txt | sacrebleu baseline/raw_data/test.en

cat bpe/translations/translation_1-5k.txt | sed 's/\@\@ //g' > bpe/translations/translation_1-5k_no_bpe.txt
sh postprocess.sh bpe/translations/translation_1-5k_no_bpe.txt bpe/translations/translation_1-5k_post.txt en
cat bpe/translations/translation_1-5k_post.txt | sacrebleu baseline/raw_data/test.en
