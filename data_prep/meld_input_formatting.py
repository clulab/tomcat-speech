# prepare MELD input for usage in networks

import os
import sys

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset

from data_prep.prepare_data import clean_up_word
from collections import OrderedDict


class MeldPrep:
    """
    A class to prepare meld for input into a generic Dataset
    """
    def __init__(self, meld_path, f_end="_IS10.csv", use_cols=None):
        self.train_path = meld_path + "/train"
        self.dev_path = meld_path + "/dev"
        self.test_path = meld_path + "/test"
        self.train = "{0}/train_sent_emo.csv".format(self.train_path)
        self.dev = "{0}/dev_sent_emo.csv".format(self.dev_path)
        self.test = "{0}/test_sent_emo.csv".format(self.test_path)

        # ordered dicts of acoustic data
        self.train_dict = OrderedDict(make_acoustic_dict("{0}/audios".format(self.train_path),
                                                         f_end=f_end, use_cols=use_cols, data_type="meld"))
        self.dev_dict = OrderedDict(make_acoustic_dict("{0}/audios".format(self.dev_path),
                                                       f_end=f_end, use_cols=use_cols, data_type="meld"))
        self.test_dict = OrderedDict(make_acoustic_dict("{0}/audios".format(self.test_path),
                                                        f_end=f_end, use_cols=use_cols, data_type="meld"))

        # utterance-level dict
        self.longest_utt, self.longest_dia = self.get_longest_utt_meld()

        # get length of longest acoustic dataframe
        self.longest_acoustic = get_max_num_acoustic_frames(list(self.train_dict.values()) +
                                                            list(self.dev_dict.values()) +
                                                            list(self.test_dict.values()))

        print(self.longest_acoustic)
        sys.exit()

        print("Finalizing acoustic organization")

    def get_longest_utt_meld(self):
        """
        Get the length of the longest utterance and dialogue in the meld
        :return: length of longest utt, length of longest dialogue
        """
        # get all data splits
        train_utts_df = pd.read_csv(self.train)
        dev_utts_df = pd.read_csv(self.dev)
        test_utts_df = pd.read_csv(self.test)

        # concatenate them and put utterances in array
        all_utts_df = pd.concat([train_utts_df, dev_utts_df, test_utts_df], axis=0)
        all_utts = all_utts_df["Utterance"].tolist()

        # get longest dialogue length
        longest_dia = max(all_utts_df['Utterance_ID'].tolist()) + 1  # because 0-indexed

        longest = get_longest_utt(all_utts)

        return longest, longest_dia


class MultitaskData(Dataset):
    """
    A dataset to manipulate the MELD data before passing to NNs
    """
    def __init__(self, data, glove, acoustic_length, data_type="meld"):
        # get the data--a dict that is the output of a Prep class
        self.data = data
        self.train = data.train
        self.dev = data.dev
        self.test = data.test
        self.train_dict = data.train_dict
        self.dev_dict = data.dev_dict
        self.test_dict = data.test_dict

        # longest utterance, dialogue, and acoustic df
        self.longest_utt = data.longest_utt
        self.longest_dia = data.longest_dia
        self.longest_acoustic = data.longest_acoustic

        # get the data type
        self.data_type = data_type

        # get the number of acoustic features
        self.acoustic_length = acoustic_length

        # Glove object
        self.glove = glove

        print("Collecting acoustic features")

        self.train_acoustic, self.train_usable_utts, self.train_acoustic_lengths = \
            self.make_acoustic_set(self.train, self.train_dict)
        self.dev_acoustic, self.dev_usable_utts, self.dev_acoustic_lengths = \
            self.make_acoustic_set(self.dev, self.dev_dict)
        self.test_acoustic, self.test_usable_utts, self.test_acoustic_lengths = \
            self.make_acoustic_set(self.test, self.test_dict)

        print("Getting text, speaker, and y features")

        # get utterance, speaker, y matrices for train, dev, and test sets
        if data_type == "meld":
            self.train_utts, self.train_spkrs, self.train_y_emo, self.train_y_sent, self.train_utt_lengths = \
                self.make_meld_data_tensors(self.train, self.train_usable_utts)

            self.dev_utts, self.dev_spkrs, self.dev_y_emo, self.dev_y_sent, self.dev_utt_lengths = \
                self.make_meld_data_tensors(self.dev, self.dev_usable_utts)

            self.test_utts, self.test_spkrs, self.test_y_emo, self.test_y_sent, self.test_utt_lengths = \
                self.make_meld_data_tensors(self.test, self.test_usable_utts)

            # set emotion and sentiment weights
            self.emotion_weights = get_class_weights(self.train_y_emo)
            self.sentiment_weights = get_class_weights(self.train_y_sent)

        elif data_type == "mustard":
            self.train_utts, self.train_spkrs, self.train_y_sarcasm, self.train_utt_lengths = \
                self.make_mustard_data_tensors(self.train)
            self.dev_utts, self.dev_spkrs, self.dev_y_sarcasm, self.dev_utt_lengths = \
                self.make_mustard_data_tensors(self.dev)
            self.test_utts, self.test_spkrs, self.test_y_sarcasm, self.test_utt_lengths = \
                self.make_mustard_data_tensors(self.test)

            # set the sarcasm weights
            self.sarcasm_weights = get_class_weights(self.train_y_sarcasm)

        # get the data organized for input into the NNs
        self.train_data, self.dev_data, self.test_data = self.combine_xs_and_ys()

        self.split = self.train_data

    def set_split(self, split):
        """
        Set the split (used for __len__)
        """
        if split == "train":
            self.split = self.train_data
        elif split == "dev":
            self.split = self.dev_data
        elif split == "test":
            self.split = self.test_data

    def __len__(self):
        return len(self.split)

    def __getitem__(self, item):
        """
        item (int) : the index to a data point
        """
        return self.split[item]

    def combine_xs_and_ys(self):
        """
        Combine all x and y data into list of tuples for easier access with DataLoader
        """
        train_data = []
        dev_data = []
        test_data = []

        for i, item in enumerate(self.train_acoustic):
            if self.data_type == "meld":
                train_data.append((item, self.train_utts[i], self.train_spkrs[i],
                                   self.train_y_emo[i], self.train_y_sent[i], self.train_utt_lengths[i],
                                   self.train_acoustic_lengths[i]))
            elif self.data_type == "mustard":
                train_data.append((item, self.train_utts[i], self.train_spkrs[i],
                                   self.train_y_sarcasm[i], self.train_utt_lengths[i],
                                   self.train_acoustic_lengths[i]))

        for i, item in enumerate(self.dev_acoustic):
            if self.data_type == "meld":
                dev_data.append((item, self.dev_utts[i], self.dev_spkrs[i],
                                 self.dev_y_emo[i], self.dev_y_sent[i], self.dev_utt_lengths[i],
                                 self.dev_acoustic_lengths[i]))
            elif self.data_type == "mustard":
                train_data.append((item, self.dev_utts[i], self.dev_spkrs[i],
                                   self.dev_y_sarcasm[i], self.dev_utt_lengths[i],
                                   self.dev_acoustic_lengths[i]))

        for i, item in enumerate(self.test_acoustic):
            if self.data_type == "meld":
                test_data.append((item, self.test_utts[i], self.test_spkrs[i],
                                  self.test_y_emo[i], self.test_y_sent[i], self.test_utt_lengths[i],
                                  self.test_acoustic_lengths[i]))
            elif self.data_type == "mustard":
                train_data.append((item, self.test_utts[i], self.test_spkrs[i],
                                   self.test_y_sarcasm[i], self.test_utt_lengths[i],
                                   self.test_acoustic_lengths[i]))

        return train_data, dev_data, test_data

    def make_meld_data_tensors(self, text_path, all_utts_list):
        """
        Prepare the tensors of utterances + speakers, emotion and sentiment scores
        :param text_path: the FULL path to a csv containing the text (in column 0)
        :param all_utts_list: a list of all usable utterances
        :return:
        """
        # create holders for the data
        all_utts = []
        all_speakers = []
        all_emotions = []
        all_sentiments = []

        # create holder for sequence lengths information
        utt_lengths = []

        all_utts_df = pd.read_csv(text_path)

        for idx, row in all_utts_df.iterrows():

            # check to make sure this utterance is used
            dia_num, utt_num = row['DiaID_UttID'].split("_")[:2]
            if (dia_num, utt_num) in all_utts_list:

                # create utterance-level holders
                emos = [0] * 7
                sents = [0] * 3
                utts = [0] * self.longest_utt

                # get values from row
                utt = row["Utterance"]
                utt = [clean_up_word(wd) for wd in utt.strip().split(" ")]
                utt_lengths.append(len(utt))

                spk_id = row['Speaker']
                emo = row['Emotion']
                sent = row['Sentiment']

                # convert words to indices for glove
                for ix, wd in enumerate(utt):
                    if wd in self.glove.wd2idx.keys():
                        utts[ix] = self.glove.wd2idx[wd]
                    else:
                        utts[ix] = self.glove.wd2idx['<UNK>']

                all_utts.append(torch.tensor(utts))
                all_speakers.append([spk_id])

                emos[emo] = 1  # assign 1 to the emotion tagged
                sents[sent] = 1  # assign 1 to the sentiment tagged
                all_emotions.append(emos)
                all_sentiments.append(sents)

        # create pytorch tensors for each
        all_speakers = torch.tensor(all_speakers)
        all_emotions = torch.tensor(all_emotions)
        all_sentiments = torch.tensor(all_sentiments)

        # padd and transpose utterance sequences
        all_utts = nn.utils.rnn.pad_sequence(all_utts)
        all_utts = all_utts.transpose(0, 1)

        # return data
        return all_utts, all_speakers, all_emotions, all_sentiments, utt_lengths

    def make_mustard_data_tensors(self, all_utts_df):
        """
        Prepare the tensors of utterances + speakers, emotion and sentiment scores
        :param all_utts_df: the dataframe of all utterances
        :return:
        """
        # create holders for the data
        all_utts = []
        all_speakers = []
        all_sarcasm = []

        # create holder for sequence lengths information
        utt_lengths = []

        for idx, row in all_utts_df.iterrows():

            # create utterance-level holders
            utts = [0] * self.longest_utt

            # get values from row
            utt = row["utterance"]
            utt = [clean_up_word(wd) for wd in utt.strip().split(" ")]
            utt_lengths.append(len(utt))

            spk_id = row['speaker']
            sarc = row['sarcasm']

            # convert words to indices for glove
            for ix, wd in enumerate(utt):
                if wd in self.glove.wd2idx.keys():
                    utts[ix] = self.glove.wd2idx[wd]
                else:
                    utts[ix] = self.glove.wd2idx['<UNK>']

            all_utts.append(torch.tensor(utts))
            all_speakers.append(spk_id)
            all_sarcasm.append([sarc])

        # get set of all speakers, create lookup dict, and get list of all speaker IDs
        speaker_set = set([speaker for speaker in all_speakers])
        speaker2idx = get_speaker_to_index_dict(speaker_set)
        speaker_ids = [[speaker2idx[speaker]] for speaker in all_speakers]

        # create pytorch tensors for each
        speaker_ids = torch.tensor(speaker_ids)
        all_sarcasm = torch.tensor(all_sarcasm)

        # padd and transpose utterance sequences
        all_utts = nn.utils.rnn.pad_sequence(all_utts)
        all_utts = all_utts.transpose(0, 1)

        # return data
        return all_utts, speaker_ids, all_sarcasm, utt_lengths

    def make_dialogue_aware_meld_data_tensors(self, text_path, all_utts_list):
        """
        Prepare the tensors of utterances + speakers, emotion and sentiment scores
        This preserves dialogue structure for use within networks
        todo: add usage of this back into the class as needed
            or (better) combine with make_meld_data_tensors
        :param text_path: the FULL path to a csv containing the text (in column 0)
        :param all_utts_list: a list of all usable utterances
        :return:
        """
        # holders for the data
        all_utts = []
        all_speakers = []
        all_emotions = []
        all_sentiments = []

        all_utts_df = pd.read_csv(text_path)
        dialogue = 0

        # dialogue-level holders
        utts = [[0] * self.longest_utt] * self.longest_dia
        spks = [0] * self.longest_dia
        emos = [[0] * 7] * self.longest_dia
        sents = [[0] * 3] * self.longest_dia

        for idx, row in all_utts_df.iterrows():

            # check to make sure this utterance is used
            dia_num, utt_num = row['DiaID_UttID'].split("_")[:2]
            if (dia_num, utt_num) in all_utts_list:

                dia_id = row['Dialogue_ID']
                utt_id = row['Utterance_ID']
                utt = row["Utterance"]
                utt = [clean_up_word(wd) for wd in utt.strip().split(" ")]

                spk_id = row['Speaker']
                emo = row['Emotion']
                sent = row['Sentiment']

                # utterance-level holder
                idxs = [0] * self.longest_utt

                # convert words to indices for glove
                for ix, wd in enumerate(utt):
                    if wd in self.glove.wd2idx.keys():
                        idxs[ix] = self.glove.wd2idx[wd]
                    else:
                        idxs[ix] = self.glove.wd2idx['<UNK>']

                if dialogue == dia_id:
                    utts[utt_id] = idxs
                    spks[utt_id] = spk_id
                    emos[utt_id][emo] = 1 # assign 1 to the emotion tagged
                    sents[utt_id][sent] = 1 # assign 1 to the sentiment tagged
                else:
                    all_utts.append(torch.tensor(utts))
                    all_speakers.append(spks)
                    all_emotions.append(emos)
                    all_sentiments.append(sents)

                    # utt_dict[dia_id] = utts
                    dialogue = dia_id

                    # dialogue-level holders
                    utts = [[0] * self.longest_utt] * self.longest_dia
                    spks = [0] * self.longest_dia
                    emos = [[0] * 7] * self.longest_dia
                    sents = [[0] * 3] * self.longest_dia

                    # utts.append(idxs)
                    utts[utt_id] = idxs
                    spks[utt_id] = spk_id
                    emos[utt_id][emo] = 1 # assign 1 to the emotion tagged
                    sents[utt_id][sent] = 1 # assign 1 to the sentiment tagged

        all_speakers = torch.tensor(all_speakers)
        all_emotions = torch.tensor(all_emotions)
        all_sentiments = torch.tensor(all_sentiments)

        all_utts = nn.utils.rnn.pad_sequence(all_utts)
        all_utts = all_utts.transpose(0, 1)

        # return data
        return all_utts, all_speakers, all_emotions, all_sentiments

    def make_acoustic_set(self, text_path, acoustic_dict):
        """
        Prep the acoustic data using the acoustic dict
        :param text_path: FULL path to file containing utterances + labels
        :param acoustic_dict:
        :return:
        """
        # read in the acoustic csv
        if type(text_path) == str:
            all_utts_df = pd.read_csv(text_path)
        elif type(text_path) == pd.core.frame.DataFrame:
            all_utts_df = text_path
        else:
            sys.exit("text_path is of unaccepted type.")
        # get lists of valid dialogues and utterances
        if self.data_type == "meld":
            valid_dia_utt = all_utts_df['DiaID_UttID'].tolist()
            valid_dia = all_utts_df['Dialogue_ID'].tolist()
            valid_utt = all_utts_df['Utterance_ID'].tolist()
        else:
            valid_dia_utt = all_utts_df['clip_id'].tolist()
            valid_utt = all_utts_df['utterance'].tolist()

        # set holders for acoustic data
        all_acoustic = []
        usable_utts = []

        acoustic_lengths = []

        # for all items with audio + gold label
        for idx, item in enumerate(valid_dia_utt):
            # if that dialogue and utterance appears has an acoustic feats file
            if (item.split("_")[0], item.split("_")[1]) in acoustic_dict.keys():
                # set intermediate acoustic holder
                intermediate_acoustic = [[0] * self.acoustic_length] * self.longest_acoustic

                # pull out the acoustic feats dataframe
                acoustic_data = acoustic_dict[(item.split("_")[0], item.split("_")[1])]

                # add this dialogue + utt combo to the list of possible ones
                usable_utts.append((item.split("_")[0], item.split("_")[1]))

                # save length of the data
                acoustic_lengths.append(len(acoustic_data))

                # add the acoustic features to the holder of features
                for i, feats in enumerate(acoustic_data):
                    intermediate_acoustic[i] = feats

                # add features as tensor to acoustic data
                all_acoustic.append(torch.tensor(intermediate_acoustic))

        # pad the sequence and reshape it to proper format
        all_acoustic = nn.utils.rnn.pad_sequence(all_acoustic)
        all_acoustic = all_acoustic.transpose(0, 1)

        return all_acoustic, usable_utts, acoustic_lengths

    def make_dialogue_aware_acoustic_set(self, text_path, acoustic_dict):
        """
        Prep the acoustic data using the acoustic dict
        :param text_path: FULL path to file containing utterances + labels
        :param acoustic_dict:
        :return:
        """
        # read in the acoustic csv
        all_utts_df = pd.read_csv(text_path)
        # get lists of valid dialogues and utterances
        valid_dia_utt = all_utts_df['DiaID_UttID'].tolist()
        valid_dia = all_utts_df['Dialogue_ID'].tolist()
        valid_utt = all_utts_df['Utterance_ID'].tolist()

        # set counter for dialogue number
        dialogue = 0

        # set holders for acoustic data
        all_acoustic = []
        usable_utts = []
        intermediate_acoustic = [[0] * self.acoustic_length] * self.longest_utt

        # for all items with audio + gold label
        for idx, item in enumerate(valid_dia_utt):
            # if that dialogue and utterance appears has an acoustic feats file
            if (item.split("_")[0], item.split("_")[1]) in acoustic_dict.keys():
                # pull out the acoustic feats dataframe
                acoustic_data = acoustic_dict[(item.split("_")[0], item.split("_")[1])]
                # add this dialogue + utt combo to the list of possible ones
                usable_utts.append((item.split("_")[0], item.split("_")[1]))

                # if the dialogue is one that's used
                if dialogue == valid_dia[idx]:
                    # get the utterance number
                    utt_num = valid_utt[idx]
                    # add the acoustic features to the holder of features
                    for i, feat in enumerate(acoustic_data):
                        intermediate_acoustic[utt_num][i] = feat
                # if dialogue isn't one that's used
                else:
                    # we know we have changed dialogues, so...
                    # add a tensor of acoustic features to the list of all
                    all_acoustic.append(torch.tensor(intermediate_acoustic))

                    # set the dialogue
                    dialogue = valid_dia[idx]

                    # zero the holder for intermediate acoustic data
                    intermediate_acoustic = [[0] * self.acoustic_length] * self.longest_utt

                    # get the utterance number
                    utt_num = valid_utt[idx]
                    # add the acoustic features to the holder of features
                    for i, feat in enumerate(acoustic_data):
                        intermediate_acoustic[utt_num][i] = feat

        # pad the sequence and reshape it to proper format
        all_acoustic = nn.utils.rnn.pad_sequence(all_acoustic)
        all_acoustic = all_acoustic.transpose(0, 1)

        return all_acoustic, usable_utts


# helper functions
def get_class_weights(y_set ):
    class_counts = {}
    if type(y_set[0]) == list or type(y_set[0]) == torch.Tensor:
        print("type of data is: " + str(type(y_set[0])))
        if len(y_set[0]) > 1:
            print("length of data[0] is: " + str(len(y_set[0])))
            y_values = [item.index(max(item)) for item in y_set.tolist()]
        else:
            print("data[0] is of length 1")
            y_values = [value for item in y_set.tolist() for value in item]
    else:
        y_values = y_set.tolist()
    print(y_values)
    for item in y_values:
        if item not in class_counts:
            class_counts[item] = 1
        else:
            class_counts[item] += 1
    class_weights = []
    for k,v in sorted(class_counts.items()):
        class_weights.append(float(v))
    class_weights = torch.tensor(class_weights)
    return 1.0 / class_weights


def get_speaker_to_index_dict(speaker_set):
    """
    Take a set of speakers and return a speaker2idx dict
    speaker_set : the set of speakers
    """
    # set counter
    speaker_num = 0

    # set speaker2idx dict
    speaker2idx = {}

    # put all speakers in
    for speaker in speaker_set:
        speaker2idx[speaker] = speaker_num
        speaker_num += 1

    return speaker2idx


def get_longest_utt(utts_list):
    """
    checks lengths of utterances in a list
    and return len of longest
    """
    longest = 0

    for utt in utts_list:
        split_utt = utt.strip().split(" ")
        utt_len = len(split_utt)
        if utt_len > longest:
            longest = utt_len

    return longest


def make_acoustic_dict(acoustic_path, files_to_get=None, f_end="_IS10.csv",
                       use_cols=None, data_type="meld", n_to_skip=2):
    """
    makes a dict of (sid, call): data for use in ClinicalDataset objects
    f_end: end of acoustic file names
    use_cols: if set, should be a list [] of column names to include
    n_to_skip : the number of columns at the start to ignore (e.g. name, time)
    """
    acoustic_dict = {}
    # find acoustic features files
    for f in os.listdir(acoustic_path):
        if f.endswith(f_end):
            if files_to_get is None or "_".join(f.split("_")[:2]) in files_to_get:

                # read in the file as a dataframe
                if use_cols is not None:
                    feats = pd.read_csv(acoustic_path + "/" + f, usecols=use_cols,
                                        sep=";")
                    feats = feats.drop(feats.columns[:n_to_skip], axis=1)
                else:
                    feats = pd.read_csv(acoustic_path + "/" + f, sep=";")
                    feats = feats.drop(feats.columns[:n_to_skip], axis=1)

                # get the dialogue and utterance IDs
                dia_id = f.split("_")[0]
                utt_id = f.split("_")[1]

                # save the dataframe to a dict with (dialogue, utt) as key
                if data_type == "meld":
                    acoustic_dict[(dia_id, utt_id)] = feats.values.tolist()[0]
                else:
                    acoustic_dict[(dia_id, utt_id)] = feats.values.tolist()

    return acoustic_dict


def get_max_num_acoustic_frames(acoustic_set):
    """
    Get the maximum number of acoustic feature frames in any utterance
    from the dataset used
    acoustic_set : the FULL set of acoustic dfs (train + dev + test)
    """
    longest = 0

    for feats_df in acoustic_set:
        utt_len = len(feats_df)
        # utt_len = feats_df.shape[0]
        if utt_len > longest:
            longest = utt_len

    return longest

