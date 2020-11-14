#! /usr/bin/python3

# I wrote this file for the Bachelor course in MT in spring semester 20, and slightly adapted it for this exercise.

import argparse
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--infile', type=str, help='path of file to read BLEU scores from', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.infile, 'r') as infile:
        file = infile.read()
        str_scores = re.findall(r'=\s(\d\d\.\d)', file)
        bleu_scores = []
        for score in str_scores:
            score = float(score)
            bleu_scores.append(score)

    beam_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    plt.grid()
    plt.scatter(beam_sizes, bleu_scores)
    plt.ylim(21, 23)
    plt.yticks(np.arange(21, 23, 0.25))
    plt.xticks(np.arange(2, 22, 2.0))
    plt.title('BLEU scores with respect to beam sizes')
    plt.xlabel('BEAM SIZE')
    plt.ylabel('BLEU')

    plt.savefig("beam_graph.png")


if __name__ == main():
    main()
