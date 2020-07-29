# prepare chalearn for input into the model

import os
import pickle
import json

import torch
from torch import nn
from torchtext.data import get_tokenizer

from data_prep.audio_extraction import ExtractAudio
import pandas as pd

from data_prep.data_prep_helpers import get_class_weights, get_gender_avgs
from data_prep.data_prep_helpers import create_data_folds_list


def convert_chalearn_pickle_to_json(path, file):
    """
    Convert the pickled data files for chalearn into json files
    """
    fname = file.split(".pkl")[0]
    pickle_file = os.path.join(path, file)
    with open(pickle_file, 'rb') as pfile:
        # use latin-1 enecoding to avoid readability issues
        data = pickle.load(pfile, encoding='latin1')

    json_file = os.path.join(path, fname + ".json")
    with open(json_file, 'w') as jfile:
        json.dump(data, jfile)


if __name__ == "__main__":
    # path = "../../datasets/multimodal_datasets/Chalearn/val"
    # file_1 = "annotation_validation.pkl"
    # file_2 = "transcription_validation.pkl"
    #
    # convert_chalearn_pickle_to_json(path, file_1)
    # convert_chalearn_pickle_to_json(path, file_2)
    pass
