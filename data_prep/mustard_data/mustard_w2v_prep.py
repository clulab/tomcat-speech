# prepare MELD input for usage in networks with wav2vec

import os
import sys

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset

from data_prep.data_prep_helpers import (
    make_w2v_dict)

from data_prep.acoustic_extraction import *

from sklearn.model_selection import train_test_split

from collections import OrderedDict, defaultdict

from torch.nn.utils.rnn import pad_sequence


class MustardPrepData(torch.utils.data.Dataset):
    def __init__(self, audio_data_path, acoustic_data_path, response_data, rnn=False):
        self.audio_path = audio_data_path
        self.acoustic_path = acoustic_data_path
        self.sentiment = {}

        with open(response_data, "r") as f:
            data = f.readlines()

        self.label_info = defaultdict(dict)

        for i in range(1, len(data)):
            items = data[i].rstrip().split("\t")
            file_id = items[0]
            utt = items[1]
            speaker = items[2]
            sarc = int(items[3])
            self.label_info[file_id]["spk"] = speaker
            self.label_info[file_id]["utt"] = utt
            self.label_info[file_id]["sarc"] = sarc

        self.wav_names = []

        self.wav_names = [name for name in list(self.label_info.keys())]

        self.audio_dict, self.audio_length = make_w2v_dict(self.audio_path, self.wav_names, rnn=False)
        self.acoustic, self.acoustic_length = read_audio(self.acoustic_path, rnn=True)

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        try:
            emotion_label = self.label_info[self.wav_names[idx]]["sarc"]
            audio_info = self.audio_dict[self.wav_names[idx]]
            acoustic_info = self.acoustic[self.wav_names[idx]]
            audio_length = self.audio_length[self.wav_names[idx]]
            acoustic_length = self.acoustic_length[self.wav_names[idx]]
            item = {'audio': audio_info,
                    'length': audio_length,
                    'acoustic': acoustic_info,
                    'acoustic_length': acoustic_length,
                    'label': emotion_label}

            return item

        except KeyError:
            next


class MustardPrep:
    """
    A class to prepare meld for input into a generic Dataset
    """

    def __init__(
            self,
            mustard_path,
            mustard_data_path,
            rnn=False
    ):
        self.path = mustard_path
        self.audio_path = os.path.join(mustard_path, "utterances_final_w2v")
        self.acoustic_path = os.path.join(mustard_path, "utterance_final_mp3")
        self.data = os.path.join(mustard_path, "mustard_utts.txv")

        self.mustard_data = os.path.join(mustard_data_path, "data.pt")

        if os.path.exists(os.path.join(mustard_data_path, "data.pt")):
            print("LOAD DATASET")
            self.dataset = torch.load(self.mustard_data)
        else:
            print("CREATING DATASET")
            self.dataset = MustardPrepData(audio_data_path=self.audio_path,
                                           acoustic_data_path=self.acoustic_path,
                                           response_data=self.data,
                                           rnn=rnn)

            with open(self.mustard_data, "wb") as data_file:
                torch.save(self.dataset, data_file)

        self.train_dataset, self.dev_dataset = train_test_split(self.dataset, test_size=0.3)

    def get_train(self):
        return self.train_dataset

    def get_dev(self):
        return self.dev_dataset
