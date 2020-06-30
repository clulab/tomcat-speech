# prepare MELD input for usage in networks

import os
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset

from data_prep.prepare_data import clean_up_word
from collections import OrderedDict


class MELDData(Dataset):
    """
    A dataset to manipulate the MELD data before passing to NNs
    """
    def __init__(self, meld_path, glove, acoustic_length, f_end="_IS10.csv", use_cols=None):
        self.train_path = meld_path + "/train"
        self.dev_path = meld_path + "/dev"
        self.test_path = meld_path + "/test"
        self.meld_train = "{0}/train_sent_emo.csv".format(self.train_path)
        self.meld_dev = "{0}/dev_sent_emo.csv".format(self.dev_path)
        self.meld_test = "{0}/test_sent_emo.csv".format(self.test_path)

        # get the number of acoustic features
        self.acoustic_length = acoustic_length

        # Glove object
        self.glove = glove

        print("Collecting acoustic features")

        # ordered dicts of acoustic data
        self.train_dict = OrderedDict(make_acoustic_dict_meld("{0}/audios".format(self.train_path),
                                                              f_end, use_cols))
        self.dev_dict = OrderedDict(make_acoustic_dict_meld("{0}/audios".format(self.dev_path),
                                                            f_end, use_cols))
        self.test_dict = OrderedDict(make_acoustic_dict_meld("{0}/audios".format(self.test_path),
                                                             f_end, use_cols))

        # utterance-level dict
        self.longest_utt, self.longest_dia = self.get_longest_utt_meld()

        print("Finalizing acoustic organization")

        self.train_acoustic, self.train_usable_utts = self.make_acoustic_set(self.meld_train, self.train_dict)
        self.dev_acoustic, self.dev_usable_utts = self.make_acoustic_set(self.meld_dev, self.dev_dict)
        self.test_acoustic, self.test_usable_utts = self.make_acoustic_set(self.meld_test, self.test_dict)

        print("Getting text, speaker, and y features")

        # get utterance, speaker, y matrices for train, dev, and test sets
        self.train_utts, self.train_spkrs, self.train_y_emo, self.train_y_sent, self.train_utt_lengths = \
            self.make_meld_data_tensors(self.meld_train, self.train_usable_utts)

        self.dev_utts, self.dev_spkrs, self.dev_y_emo, self.dev_y_sent, self.dev_utt_lengths = \
            self.make_meld_data_tensors(self.meld_dev, self.dev_usable_utts)

        self.test_utts, self.test_spkrs, self.test_y_emo, self.test_y_sent, self.test_utt_lengths = \
            self.make_meld_data_tensors(self.meld_test, self.test_usable_utts)

        # get the data organized for input into the NNs
        self.train_data, self.dev_data, self.test_data = self.combine_xs_and_ys()

        self.emotion_weights = get_class_weights(self.train_y_emo)

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
            train_data.append((item, self.train_utts[i], self.train_spkrs[i],
                               self.train_y_emo[i], self.train_y_sent[i], self.train_utt_lengths[i]))

        for i, item in enumerate(self.dev_acoustic):
            dev_data.append((item, self.dev_utts[i], self.dev_spkrs[i],
                             self.dev_y_emo[i], self.dev_y_sent[i], self.dev_utt_lengths[i]))

        for i, item in enumerate(self.test_acoustic):
            test_data.append((item, self.test_utts[i], self.test_spkrs[i],
                              self.test_y_emo[i], self.test_y_sent[i], self.test_utt_lengths[i]))

        return train_data, dev_data, test_data

    def get_longest_utt_meld(self):
        """
        Get the length of the longest utterance and dialogue in the meld
        :return: length of longest utt, length of longest dialogue
        """
        longest = 0

        # get all data splits
        train_utts_df = pd.read_csv(self.meld_train)
        dev_utts_df = pd.read_csv(self.meld_dev)
        test_utts_df = pd.read_csv(self.meld_test)

        # concatenate them and put utterances in array
        all_utts_df = pd.concat([train_utts_df, dev_utts_df, test_utts_df], axis=0)
        all_utts = all_utts_df["Utterance"].tolist()

        # get longest dialogue length
        longest_dia = max(all_utts_df['Utterance_ID'].tolist()) + 1  # because 0-indexed

        # check lengths and return len of longest
        for utt in all_utts:
            split_utt = utt.strip().split(" ")
            utt_len = len(split_utt)
            if utt_len > longest:
                longest = utt_len

        return longest, longest_dia

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
def get_class_weights(y_set):
    class_counts = {}
    y_values = [item.index(max(item)) for item in y_set.tolist()]
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


def make_acoustic_dict_meld(acoustic_path, f_end="_IS10.csv", use_cols=None):
    """
    makes a dict of (sid, call): data for use in ClinicalDataset objects
    f_end: end of acoustic file names
    use_cols: if set, should be a list [] of column names to include
    """
    acoustic_dict = {}
    # find acoustic features files
    for f in os.listdir(acoustic_path):
        if f.endswith(f_end):
            # read in the file as a dataframe
            if use_cols is not None:
                feats = pd.read_csv(acoustic_path + "/" + f, usecols=use_cols)
            else:
                feats = pd.read_csv(acoustic_path + "/" + f)

            # get the dialogue and utterance IDs
            dia_id = f.split("_")[0]
            utt_id = f.split("_")[1]

            # save the dataframe to a dict with (dialogue, utt) as key
            acoustic_dict[(dia_id, utt_id)] = feats.values.tolist()[0]

    return acoustic_dict
