# prepare MELD input for usage in networks with wav2vec

import os
import sys

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset

from data_prep.data_prep_helpers import (
    make_w2v_dict)

from collections import OrderedDict, defaultdict

from torch.nn.utils.rnn import pad_sequence


def emotion_to_int(emotion):
    if emotion == "neutral":
        return 0
    elif emotion == "disgust":
        return 1
    elif emotion == "fear":
        return 2
    elif emotion == "joy":
        return 3
    elif emotion == "anger":
        return 4
    elif emotion == "sadness":
        return 5
    else:
        return 6


def sentiment_to_int(sentiment):
    if sentiment == "neutral":
        return 0
    elif sentiment == "positive":
        return 1
    else:
        return 2


class MeldPrepData(torch.utils.data.Dataset):
    def __init__(self, audio_data_path, response_data):
        self.audio_path = audio_data_path
        self.sentiment = {}

        with open(response_data, "r") as f:
            data = f.readlines()

        self.label_info = defaultdict(dict)

        for i in range(1, len(data)):
            items = data[i].rstrip().split("\t")
            dia_id = items[5]
            utt_id = items[6]
            file_id = "dia%s_utt%s" % (dia_id, utt_id)
            speaker = items[2]
            emotion = items[3]
            sentiment = items[4]
            self.label_info[file_id]["spk"] = speaker
            self.label_info[file_id]["emot"] = emotion_to_int(emotion)
            self.label_info[file_id]["sent"] = sentiment_to_int(sentiment)

        self.wav_names = []

        self.wav_names = [name for name in list(self.label_info.keys())]

        self.audio_dict, self.audio_length = make_w2v_dict(self.audio_path, self.wav_names, rnn=True)

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        try:
            emotion_label = self.label_info[self.wav_names[idx]]["emot"]
            audio_info = self.audio_dict[self.wav_names[idx]]
            audio_length = self.audio_length[self.wav_names[idx]]
            item = {'audio': audio_info, 'length': audio_length, 'label': emotion_label}

            return item

        except KeyError:
            next


class MeldPrep:
    """
    A class to prepare meld for input into a generic Dataset
    """

    def __init__(
            self,
            meld_path,
            meld_data_path,
            wav_model
    ):
        self.path = meld_path
        self.audio_path = meld_path + "/meld_data"
        # self.train_path = meld_path + "/train"
        # self.dev_path = meld_path + "/dev"
        # self.test_path = meld_path + "/test"
        self.train = "{0}/train_sent_emo.csv".format(meld_path)
        self.dev = "{0}/dev_sent_emo.csv".format(meld_path)
        self.test = "{0}/test_sent_emo.csv".format(meld_path)

        self.meld_train_data = os.path.join(meld_data_path, "train.pt")
        self.meld_dev_data = os.path.join(meld_data_path, "dev.pt")
        self.meld_test_data = os.path.join(meld_data_path, "test.pt")

        print(self.meld_train_data)
        print(self.meld_dev_data)
        print(self.meld_test_data)

        if os.path.exists(os.path.join(meld_data_path, "train.pt")):
            print("LOAD DATASET")
            self.train_dataset = torch.load(self.meld_train_data)
            self.dev_dataset = torch.load(self.meld_dev_data)
            self.test_dataset = torch.load(self.meld_test_data)
        else:
            print("CREATING DATASET")
            self.train_dataset = MeldPrepData(audio_path=self.audio_path,
                                              response_data=self.train)

            with open(os.path.join(self.meld_train_data), "wb") as data_file:
                torch.save(self.train_dataset, data_file)

            self.dev_dataset = MeldPrepData(audio_path=self.audio_path,
                                            response_data=self.dev)

            with open(os.path.join(meld_data_path, "dev.pt"), "wb") as data_file:
                torch.save(self.dev_dataset, data_file)

            self.test_dataset = MeldPrepData(audio_path=self.audio_path,
                                             response_data=self.test)

            with open(os.path.join(meld_data_path, "test.pt"), "wb") as data_file:
                torch.save(self.test_dataset, data_file)

    def get_train(self):
        return self.train_dataset

    def get_dev(self):
        return self.dev_dataset

    def get_test(self):
        return self.test_dataset


