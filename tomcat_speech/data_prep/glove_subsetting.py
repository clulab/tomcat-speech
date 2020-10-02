# get the subset of GloVe that relates to the vocabulary present in the texts
# this should allow for faster usage later

import os
import sys

import pandas as pd
import numpy as np
from tomcat_speech.data_prep.data_prep_helpers import clean_up_word


def get_all_vocab(data_dir):
    """
    Get all the words in the vocabulary from a given directory
    :param data_dir:
    :return:
    """
    # save to set
    all_vocab = set()
    #
    for f in os.listdir(data_dir):
        if f.endswith("IS09_avgd.csv"):
            wds = pd.read_csv(data_dir + "/" + f, usecols=["word"])
            wds = wds["word"].tolist()
            for item in wds:
                item = clean_up_word(item)
                all_vocab.add(item)
    return all_vocab


def subset_glove(glove_path, vocab_set, vec_len=100, add_unk=True):
    with open(glove_path, "r") as glove:
        subset = []
        num_items = 0
        if add_unk:
            unk_vec = np.zeros(vec_len)
        for line in glove:
            vals = line.split()
            if vals[0] in vocab_set:
                num_items += 1
                subset.append(vals)
                if add_unk:
                    unk_vec = unk_vec + np.array([float(item) for item in vals[1:]])
    if add_unk:
        unk_vec = unk_vec / num_items
        unk_vec = ["<UNK>"] + unk_vec.tolist()
        unk_vec = [str(item) for item in unk_vec]
        subset.append(unk_vec)
    return subset


def save_subset(subset, save_path):
    with open(save_path, "w") as gfile:
        for item in subset:
            gfile.write(" ".join(item))
            gfile.write("\n")
