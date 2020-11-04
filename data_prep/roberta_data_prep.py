### Generate dataset for AudioOnly Roberta

import torch
import os
from collections import defaultdict
import pickle
import numpy as np

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


class AudioOnlyData(torch.utils.data.Dataset):
    def __init__(self, audio_path, audio_token_path, response_data):
        self.audio_path = audio_path
        self.audio_token = audio_token_path
        # with open(length_data, "rb") as length_dict:
        #     self.length_data = pickle.load(length_dict)
        # print(self.length_data)
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

        self.wav_names = [name.replace(".wav", "") for name in os.listdir("data/audio_train")]

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        audio_token_name = os.path.join(self.audio_token, self.wav_names[idx] + ".txt")
        at = torch.as_tensor(np.genfromtxt(audio_token_name, delimiter="\t", dtype=np.int32)[:-1])
        emotion_label = self.label_info[self.wav_names[idx]]["emot"]

        # audio_length = self.length_data[self.wav_names[idx]][0]

        item = {'token': at, 'label': emotion_label}

        return item

