#!/bin/bash

mkdir beam_translations
mkdir beam_translations/pre
mkdir beam_translations/post

# make translations with beam sizes from 2 to 20
python3 translate_beam.py --cuda True --output beam_translations/pre/translation_beam2.txt --beam-size 2
python3 translate_beam.py --cuda True --output beam_translations/pre/translation_beam4.txt --beam-size 4
python3 translate_beam.py --cuda True --output beam_translations/pre/translation_beam6.txt --beam-size 6
python3 translate_beam.py --cuda True --output beam_translations/pre/translation_beam8.txt --beam-size 8
python3 translate_beam.py --cuda True --output beam_translations/pre/translation_beam10.txt --beam-size 10
python3 translate_beam.py --cuda True --output beam_translations/pre/translation_beam12.txt --beam-size 12
python3 translate_beam.py --cuda True --output beam_translations/pre/translation_beam14.txt --beam-size 14
python3 translate_beam.py --cuda True --output beam_translations/pre/translation_beam16.txt --beam-size 16
python3 translate_beam.py --cuda True --output beam_translations/pre/translation_beam18.txt --beam-size 18
python3 translate_beam.py --cuda True --output beam_translations/pre/translation_beam20.txt --beam-size 20

# postprecess all translations
sh postprocess_asg4.sh beam_translations/pre/translation_beam2.txt beam_translations/post/translation_beam2.txt en
sh postprocess_asg4.sh beam_translations/pre/translation_beam4.txt beam_translations/post/translation_beam4.txt en
sh postprocess_asg4.sh beam_translations/pre/translation_beam6.txt beam_translations/post/translation_beam6.txt en
sh postprocess_asg4.sh beam_translations/pre/translation_beam8.txt beam_translations/post/translation_beam8.txt en
sh postprocess_asg4.sh beam_translations/pre/translation_beam10.txt beam_translations/post/translation_beam10.txt en
sh postprocess_asg4.sh beam_translations/pre/translation_beam12.txt beam_translations/post/translation_beam12.txt en
sh postprocess_asg4.sh beam_translations/pre/translation_beam14.txt beam_translations/post/translation_beam14.txt en
sh postprocess_asg4.sh beam_translations/pre/translation_beam16.txt beam_translations/post/translation_beam16.txt en
sh postprocess_asg4.sh beam_translations/pre/translation_beam18.txt beam_translations/post/translation_beam18.txt en
sh postprocess_asg4.sh beam_translations/pre/translation_beam20.txt beam_translations/post/translation_beam20.txt en

# calculate BLEU scores and write results to a file
cat beam_translations/post/translation_beam2.txt | sacrebleu data_asg4/raw_data/test.en > beam_translations/beam_size_results.txt
cat beam_translations/post/translation_beam4.txt | sacrebleu data_asg4/raw_data/test.en >> beam_translations/beam_size_results.txt
cat beam_translations/post/translation_beam6.txt | sacrebleu data_asg4/raw_data/test.en >> beam_translations/beam_size_results.txt
cat beam_translations/post/translation_beam8.txt | sacrebleu data_asg4/raw_data/test.en >> beam_translations/beam_size_results.txt
cat beam_translations/post/translation_beam10.txt | sacrebleu data_asg4/raw_data/test.en >> beam_translations/beam_size_results.txt
cat beam_translations/post/translation_beam12.txt | sacrebleu data_asg4/raw_data/test.en >> beam_translations/beam_size_results.txt
cat beam_translations/post/translation_beam14.txt | sacrebleu data_asg4/raw_data/test.en >> beam_translations/beam_size_results.txt
cat beam_translations/post/translation_beam16.txt | sacrebleu data_asg4/raw_data/test.en >> beam_translations/beam_size_results.txt
cat beam_translations/post/translation_beam18.txt | sacrebleu data_asg4/raw_data/test.en >> beam_translations/beam_size_results.txt
cat beam_translations/post/translation_beam20.txt | sacrebleu data_asg4/raw_data/test.en >> beam_translations/beam_size_results.txt

# I wrote the file beam_graph.py for the Bachelor course in MT in spring semester 20, and slightly adapted it for this exercise.
# This reads all BLEU scores from the file we wrote above and plots the BLEU scores for each beam size
python3 beam_graph.py --infile beam_translations/beam_size_results.txt
