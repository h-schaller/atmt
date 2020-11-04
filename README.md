# ATMT Assignemt 3
The **test set translations** (post-processed) can be found in the repository’s folders
- `bpe/translations`
- `dropout/translations`
- `combined_bpe_dropout/translations`.

To recreate all models and evaluate them, it is sufficient to run these shell scripts in the command line:
- Train models with different BPE vocabulary sizes: `sh preprocess_bpe_data_and_train_nmt_models.sh`
- Train models with different dropout rates: `sh train_nmt_models_with_different_dropouts.sh`
- Train a model that combines the best results of the models above: `sh train_combined_bpe_dropout.sh`
