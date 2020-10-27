mkdir bpe
mkdir bpe/preprocessed_data
mkdir bpe/prepared_data
mkdir bpe/models

# learn BPE model with vocabulary size 6'000
subword-nmt learn-joint-bpe-and-vocab --input baseline/preprocessed_data/train.de baseline/preprocessed_data/train.en --write-vocabulary bpe/prepared_data/dict_6k.de bpe/prepared_data/dict_6k.en --symbols 6000 --total-symbols --output bpe/models/bpe_model_6k

# apply BPE model to all splits (train, tiny_train, validation, test) of datasets of source (de) and target languages (en)

# source texts
subword-nmt apply-bpe --codes bpe/models/bpe_model_6k --vocabulary bpe/prepared_data/dict_6k.de --vocabulary-threshold 1 < baseline/preprocessed_data/train.de > bpe/preprocessed_data/train_6k.de
subword-nmt apply-bpe --codes bpe/models/bpe_model_6k --vocabulary bpe/prepared_data/dict_6k.de --vocabulary-threshold 1 < baseline/preprocessed_data/test.de > bpe/preprocessed_data/test_6k.de
subword-nmt apply-bpe --codes bpe/models/bpe_model_6k --vocabulary bpe/prepared_data/dict_6k.de --vocabulary-threshold 1 < baseline/preprocessed_data/tiny_train.de > bpe/preprocessed_data/tiny_train_6k.de
subword-nmt apply-bpe --codes bpe/models/bpe_model_6k --vocabulary bpe/prepared_data/dict_6k.de --vocabulary-threshold 1 < baseline/preprocessed_data/valid.de > bpe/preprocessed_data/valid_6k.de

# target texts
subword-nmt apply-bpe --codes bpe/models/bpe_model_6k --vocabulary bpe/prepared_data/dict_6k.en --vocabulary-threshold 1 < baseline/preprocessed_data/train.en > bpe/preprocessed_data/train_6k.en
subword-nmt apply-bpe --codes bpe/models/bpe_model_6k --vocabulary bpe/prepared_data/dict_6k.en --vocabulary-threshold 1 < baseline/preprocessed_data/test.en > bpe/preprocessed_data/test_6k.en
subword-nmt apply-bpe --codes bpe/models/bpe_model_6k --vocabulary bpe/prepared_data/dict_6k.en --vocabulary-threshold 1 < baseline/preprocessed_data/tiny_train.en > bpe/preprocessed_data/tiny_train_6k.en
subword-nmt apply-bpe --codes bpe/models/bpe_model_6k --vocabulary bpe/prepared_data/dict_6k.en --vocabulary-threshold 1 < baseline/preprocessed_data/valid.en > bpe/preprocessed_data/valid_6k.en

###

# learn BPE model with vocabulary size 1'500
subword-nmt learn-joint-bpe-and-vocab --input baseline/preprocessed_data/train.de baseline/preprocessed_data/train.en --write-vocabulary bpe/prepared_data/dict_1-5k.de bpe/prepared_data/dict_1-5k.en --symbols 1500 --total-symbols --output bpe/models/bpe_model_1-5k

# apply BPE model to all splits (train, tiny_train, validation, test) of datasets of source (de) and target languages (en)

# source texts
subword-nmt apply-bpe --codes bpe/models/bpe_model_1-5k --vocabulary bpe/prepared_data/dict_1-5k.de --vocabulary-threshold 1 < baseline/preprocessed_data/train.de > bpe/preprocessed_data/train_1-5k.de
subword-nmt apply-bpe --codes bpe/models/bpe_model_1-5k --vocabulary bpe/prepared_data/dict_1-5k.de --vocabulary-threshold 1 < baseline/preprocessed_data/test.de > bpe/preprocessed_data/test_1-5k.de
subword-nmt apply-bpe --codes bpe/models/bpe_model_1-5k --vocabulary bpe/prepared_data/dict_1-5k.de --vocabulary-threshold 1 < baseline/preprocessed_data/tiny_train.de > bpe/preprocessed_data/tiny_train_1-5k.de
subword-nmt apply-bpe --codes bpe/models/bpe_model_1-5k --vocabulary bpe/prepared_data/dict_1-5k.de --vocabulary-threshold 1 < baseline/preprocessed_data/valid.de > bpe/preprocessed_data/valid_1-5k.de

# target texts
subword-nmt apply-bpe --codes bpe/models/bpe_model_1-5k --vocabulary bpe/prepared_data/dict_1-5k.en --vocabulary-threshold 1 < baseline/preprocessed_data/train.en > bpe/preprocessed_data/train_1-5k.en
subword-nmt apply-bpe --codes bpe/models/bpe_model_1-5k --vocabulary bpe/prepared_data/dict_1-5k.en --vocabulary-threshold 1 < baseline/preprocessed_data/test.en > bpe/preprocessed_data/test_1-5k.en
subword-nmt apply-bpe --codes bpe/models/bpe_model_1-5k --vocabulary bpe/prepared_data/dict_1-5k.en --vocabulary-threshold 1 < baseline/preprocessed_data/tiny_train.en > bpe/preprocessed_data/tiny_train_1-5k.en
subword-nmt apply-bpe --codes bpe/models/bpe_model_1-5k --vocabulary bpe/prepared_data/dict_1-5k.en --vocabulary-threshold 1 < baseline/preprocessed_data/valid.en > bpe/preprocessed_data/valid_1-5k.en