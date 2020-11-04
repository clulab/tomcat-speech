# prepare MELD input for usage in networks

import os

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset

from tomcat_speech.data_prep.data_prep_helpers import (
    clean_up_word,
    get_speaker_to_index_dict,
    get_max_num_acoustic_frames,
    get_speaker_gender,
    get_class_weights,
    get_gender_avgs,
    make_acoustic_set,
    transform_acoustic_item,
)
from collections import OrderedDict

import torchtext
from torchtext.data import get_tokenizer
from torch.nn.utils.rnn import pad_sequence


class MeldPrep:
    """
    A class to prepare meld for input into a generic Dataset
    """

    def __init__(
        self,
        meld_path,
        acoustic_length,
        glove,
        f_end="_IS10.csv",
        use_cols=None,
        add_avging=True,
        avgd=False,
    ):
        self.path = meld_path
        self.train_path = meld_path + "/train"
        self.dev_path = meld_path + "/dev"
        self.test_path = meld_path + "/test"
        self.train = f"{self.train_path}/train_sent_emo.csv"
        self.dev = f"{self.dev_path}/dev_sent_emo.csv"
        self.test = f"{self.test_path}/test_sent_emo.csv"

        # get files containing gold labels/data
        self.train_data_file = pd.read_csv(self.train)
        self.dev_data_file = pd.read_csv(self.dev)
        self.test_data_file = pd.read_csv(self.test)

        # get tokenizer
        self.tokenizer = get_tokenizer("basic_english")

        # get the number of acoustic features
        self.acoustic_length = acoustic_length

        # get gender of speakers
        self.speaker2gender = get_speaker_gender(
            f"{self.path}/speaker2idx.csv"
        )

        # to determine whether incoming acoustic features are averaged
        self.avgd = avgd

        if avgd:
            self.avgd = True
            self.train_dir = "audios"
            self.dev_dir = "audios"
            self.test_dir = "audios"
        else:
            self.avgd = False
            self.train_dir = "IS10_train"
            self.dev_dir = "IS10_dev"
            self.test_dir = "IS10_test"

        print("Collecting acoustic features")

        # ordered dicts of acoustic data
        # todo: is this screwed up?
        #   ordered dict--is it same order as acoustic_lengths
        self.train_dict, self.train_acoustic_lengths = make_acoustic_dict_meld(
            "{0}/{1}".format(self.train_path, self.train_dir),
            f_end,
            use_cols=use_cols,
            avgd=avgd,
        )
        self.train_dict = OrderedDict(self.train_dict)
        self.dev_dict, self.dev_acoustic_lengths = make_acoustic_dict_meld(
            "{0}/{1}".format(self.dev_path, self.dev_dir),
            f_end,
            use_cols=use_cols,
            avgd=avgd,
        )
        self.dev_dict = OrderedDict(self.dev_dict)
        self.test_dict, self.test_acoustic_lengths = make_acoustic_dict_meld(
            "{0}/{1}".format(self.test_path, self.test_dir),
            f_end,
            use_cols=use_cols,
            avgd=avgd,
        )
        self.test_dict = OrderedDict(self.test_dict)

        # utterance-level dict
        self.longest_utt, self.longest_dia = self.get_longest_utt_meld()

        # get length of longest acoustic dataframe
        self.longest_acoustic = get_max_num_acoustic_frames(
            list(self.train_dict.values())
            + list(self.dev_dict.values())
            + list(self.test_dict.values())
        )

        print("Finalizing acoustic organization")

        self.train_acoustic, self.train_usable_utts = make_acoustic_set(
            self.train,
            self.train_dict,
            data_type="meld",
            acoustic_length=acoustic_length,
            longest_acoustic=self.longest_acoustic,
            add_avging=add_avging,
            avgd=avgd,
        )
        self.dev_acoustic, self.dev_usable_utts = make_acoustic_set(
            self.dev,
            self.dev_dict,
            data_type="meld",
            acoustic_length=acoustic_length,
            longest_acoustic=self.longest_acoustic,
            add_avging=add_avging,
            avgd=avgd,
        )
        self.test_acoustic, self.test_usable_utts = make_acoustic_set(
            self.test,
            self.test_dict,
            data_type="meld",
            acoustic_length=acoustic_length,
            longest_acoustic=self.longest_acoustic,
            add_avging=add_avging,
            avgd=avgd,
        )

        # get utterance, speaker, y matrices for train, dev, and test sets
        (
            self.train_utts,
            self.train_spkrs,
            self.train_genders,
            self.train_y_emo,
            self.train_y_sent,
            self.train_utt_lengths,
        ) = self.make_meld_data_tensors(
            self.train_data_file, self.train_usable_utts, glove
        )

        (
            self.dev_utts,
            self.dev_spkrs,
            self.dev_genders,
            self.dev_y_emo,
            self.dev_y_sent,
            self.dev_utt_lengths,
        ) = self.make_meld_data_tensors(
            self.dev_data_file, self.dev_usable_utts, glove
        )

        (
            self.test_utts,
            self.test_spkrs,
            self.test_genders,
            self.test_y_emo,
            self.test_y_sent,
            self.test_utt_lengths,
        ) = self.make_meld_data_tensors(
            self.test_data_file, self.test_usable_utts, glove
        )

        # set emotion and sentiment weights
        self.emotion_weights = get_class_weights(self.train_y_emo)
        self.sentiment_weights = get_class_weights(self.train_y_sent)

        # acoustic feature normalization based on train
        self.all_acoustic_means = self.train_acoustic.mean(
            dim=0, keepdim=False
        )
        self.all_acoustic_deviations = self.train_acoustic.std(
            dim=0, keepdim=False
        )

        self.male_acoustic_means, self.male_deviations = get_gender_avgs(
            self.train_acoustic, self.train_genders, gender=2
        )
        self.female_acoustic_means, self.female_deviations = get_gender_avgs(
            self.train_acoustic, self.train_genders, gender=1
        )

        # get the data organized for input into the NNs
        (
            self.train_data,
            self.dev_data,
            self.test_data,
        ) = self.combine_xs_and_ys()

    def combine_xs_and_ys(self):
        """
        Combine all x and y data into list of tuples for easier access with DataLoader
        """
        train_data = []
        dev_data = []
        test_data = []

        for i, item in enumerate(self.train_acoustic):
            # normalize
            if self.train_genders[i] == 0:
                item_transformed = transform_acoustic_item(
                    item, self.all_acoustic_means, self.all_acoustic_deviations
                )
            elif self.train_genders[i] == 1:
                item_transformed = transform_acoustic_item(
                    item, self.female_acoustic_means, self.female_deviations
                )
            else:
                item_transformed = transform_acoustic_item(
                    item, self.male_acoustic_means, self.male_deviations
                )
            train_data.append(
                (
                    item_transformed,
                    self.train_utts[i],
                    self.train_spkrs[i],
                    self.train_genders[i],
                    self.train_y_emo[i],
                    self.train_y_sent[i],
                    self.train_utt_lengths[i],
                    self.train_acoustic_lengths[i],
                )
            )

        for i, item in enumerate(self.dev_acoustic):
            if self.train_genders[i] == 0:
                item_transformed = transform_acoustic_item(
                    item, self.all_acoustic_means, self.all_acoustic_deviations
                )
            elif self.train_genders[i] == 1:
                item_transformed = transform_acoustic_item(
                    item, self.female_acoustic_means, self.female_deviations
                )
            else:
                item_transformed = transform_acoustic_item(
                    item, self.male_acoustic_means, self.male_deviations
                )
            dev_data.append(
                (
                    item_transformed,
                    self.dev_utts[i],
                    self.dev_spkrs[i],
                    self.dev_genders[i],
                    self.dev_y_emo[i],
                    self.dev_y_sent[i],
                    self.dev_utt_lengths[i],
                    self.dev_acoustic_lengths[i],
                )
            )

        for i, item in enumerate(self.test_acoustic):
            if self.train_genders[i] == 0:
                item_transformed = transform_acoustic_item(
                    item, self.all_acoustic_means, self.all_acoustic_deviations
                )
            elif self.train_genders[i] == 1:
                item_transformed = transform_acoustic_item(
                    item, self.female_acoustic_means, self.female_deviations
                )
            else:
                item_transformed = transform_acoustic_item(
                    item, self.male_acoustic_means, self.male_deviations
                )
            test_data.append(
                (
                    item_transformed,
                    self.test_utts[i],
                    self.test_spkrs[i],
                    self.test_genders[i],
                    self.test_y_emo[i],
                    self.test_y_sent[i],
                    self.test_utt_lengths[i],
                    self.test_acoustic_lengths[i],
                )
            )

        return train_data, dev_data, test_data

    def get_longest_utt_meld(self):
        """
        Get the length of the longest utterance and dialogue in the meld
        :return: length of longest utt, length of longest dialogue
        """
        longest = 0

        # get all data splits
        train_utts_df = self.train_data_file
        dev_utts_df = self.dev_data_file
        test_utts_df = self.test_data_file

        # concatenate them and put utterances in array
        all_utts_df = pd.concat(
            [train_utts_df, dev_utts_df, test_utts_df], axis=0
        )
        all_utts = all_utts_df["Utterance"].tolist()

        for i, item in enumerate(all_utts):
            item = clean_up_word(item)
            item = self.tokenizer(item)
            if len(item) > longest:
                longest = len(item)

        # get longest dialogue length
        longest_dia = (
            max(all_utts_df["Utterance_ID"].tolist()) + 1
        )  # because 0-indexed

        return longest, longest_dia

    def make_meld_data_tensors(self, all_utts_df, all_utts_list, glove):
        """
        Prepare the tensors of utterances + speakers, emotion and sentiment scores
        :param all_utts_df: the df containing the text (in column 0)
        :param all_utts_list: a list of all usable utterances
        :param glove: an instance of class Glove
        :return:
        """
        # create holders for the data
        all_utts = []
        all_speakers = []
        all_genders = []
        all_emotions = []
        all_sentiments = []

        # create holder for sequence lengths information
        utt_lengths = []

        for idx, row in all_utts_df.iterrows():

            # check to make sure this utterance is used
            dia_num, utt_num = row["DiaID_UttID"].split("_")[:2]
            if (dia_num, utt_num) in all_utts_list:

                # create utterance-level holders
                utts = [0] * self.longest_utt

                # get values from row
                utt = clean_up_word(row["Utterance"])
                utt = self.tokenizer(utt)
                utt_lengths.append(len(utt))

                spk_id = row["Speaker"]
                gen = self.speaker2gender[spk_id]
                emo = row["Emotion"]
                sent = row["Sentiment"]

                # convert words to indices for glove
                utt_indexed = glove.index(utt)
                for i, item in enumerate(utt_indexed):
                    utts[i] = item

                all_utts.append(torch.tensor(utts))
                all_speakers.append([spk_id])
                all_genders.append(gen)
                all_emotions.append(emo)
                all_sentiments.append(sent)

        # create pytorch tensors for each
        all_speakers = torch.tensor(all_speakers)
        all_genders = torch.tensor(all_genders)
        all_emotions = torch.tensor(all_emotions)
        all_sentiments = torch.tensor(all_sentiments)

        # return data
        return (
            all_utts,
            all_speakers,
            all_genders,
            all_emotions,
            all_sentiments,
            utt_lengths,
        )

    def make_dialogue_aware_meld_data_tensors(
        self, text_path, all_utts_list, glove
    ):
        """
        Prepare the tensors of utterances + speakers, emotion and sentiment scores
        This preserves dialogue structure for use within networks
        todo: add usage of this back into the class as needed
            or (better) combine with make_meld_data_tensors
        :param text_path: the FULL path to a csv containing the text (in column 0)
        :param all_utts_list: a list of all usable utterances
        :param glove: an instance of class Glove
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
            dia_num, utt_num = row["DiaID_UttID"].split("_")[:2]
            if (dia_num, utt_num) in all_utts_list:

                dia_id = row["Dialogue_ID"]
                utt_id = row["Utterance_ID"]
                utt = row["Utterance"]
                utt = [clean_up_word(wd) for wd in utt.strip().split(" ")]

                spk_id = row["Speaker"]
                emo = row["Emotion"]
                sent = row["Sentiment"]

                # utterance-level holder
                idxs = [0] * self.longest_utt

                # convert words to indices for glove
                for ix, wd in enumerate(utt):
                    if wd in glove.wd2idx.keys():
                        idxs[ix] = glove.wd2idx[wd]
                    else:
                        idxs[ix] = glove.wd2idx["<UNK>"]

                if dialogue == dia_id:
                    utts[utt_id] = idxs
                    spks[utt_id] = spk_id
                    emos[utt_id][emo] = 1  # assign 1 to the emotion tagged
                    sents[utt_id][sent] = 1  # assign 1 to the sentiment tagged
                else:
                    all_utts.append(torch.tensor(utts))
                    all_speakers.append(spks)
                    all_emotions.append(emos)
                    all_sentiments.append(sents)

                    dialogue = dia_id

                    # dialogue-level holders
                    utts = [[0] * self.longest_utt] * self.longest_dia
                    spks = [0] * self.longest_dia
                    emos = [[0] * 7] * self.longest_dia
                    sents = [[0] * 3] * self.longest_dia

                    utts[utt_id] = idxs
                    spks[utt_id] = spk_id
                    emos[utt_id][emo] = 1  # assign 1 to the emotion tagged
                    sents[utt_id][sent] = 1  # assign 1 to the sentiment tagged

        all_speakers = torch.tensor(all_speakers)
        all_emotions = torch.tensor(all_emotions)
        all_sentiments = torch.tensor(all_sentiments)

        # return data
        return all_utts, all_speakers, all_emotions, all_sentiments


# helper functions
def make_acoustic_dict_meld(
    acoustic_path,
    f_end="_IS10.csv",
    files_to_get=None,
    use_cols=None,
    avgd=True,
):
    """
    makes a dict of (dia, utt): data for use in MELD objects
    f_end: end of acoustic file names
    use_cols: if set, should be a list [] of column names to include
    n_to_skip : the number of columns at the start to ignore (e.g. name, time)
    """
    acoustic_dict = {}
    # acoustic_lengths = []
    acoustic_lengths = {}
    # find acoustic features files
    for f in os.listdir(acoustic_path):
        if f.endswith(f_end):
            # set the separator--averaged files are actually CSV, others are ;SV
            if avgd:
                separator = ","
            else:
                separator = ";"

            # read in the file as a dataframe
            if (
                files_to_get is None
                or "_".join(f.split("_")[:2]) in files_to_get
            ):
                if use_cols is not None:
                    feats = pd.read_csv(
                        acoustic_path + "/" + f,
                        usecols=use_cols,
                        sep=separator,
                    )
                else:
                    feats = pd.read_csv(acoustic_path + "/" + f, sep=separator)
                    if not avgd:
                        feats.drop(["name", "frameTime"], axis=1, inplace=True)

                # get the dialogue and utterance IDs
                dia_id = f.split("_")[0]
                utt_id = f.split("_")[1]

                # save the dataframe to a dict with (dialogue, utt) as key
                if feats.shape[0] > 0:
                    acoustic_dict[(dia_id, utt_id)] = feats.values.tolist()
                    acoustic_lengths[(dia_id, utt_id)] = feats.shape[0]

    # sort acoustic lengths so they are in the same order as other data
    acoustic_lengths = [
        value for key, value in sorted(acoustic_lengths.items())
    ]

    return acoustic_dict, acoustic_lengths


# def make_dialogue_aware_acoustic_set(
#     text_path,
#     acoustic_dict,
#     data_type,
#     acoustic_length,
#     longest_acoustic,
#     add_avging=True,
#     avgd=False,
# ):
#     """
#     Prep the acoustic data using the acoustic dict
#     :param text_path: FULL path to file containing utterances + labels
#     :param acoustic_dict:
#     :param add_avging:
#     :return:
#     """
#     # read in the acoustic csv
#     all_utts_df = pd.read_csv(text_path)
#     # get lists of valid dialogues and utterances
#     valid_dia_utt = all_utts_df["DiaID_UttID"].tolist()
#
#     # set holders for acoustic data
#     all_acoustic = []
#     usable_utts = []
#
#     # for all items with audio + gold label
#     for idx, item in enumerate(valid_dia_utt):
#         # if that dialogue and utterance appears has an acoustic feats file
#         if (item.split("_")[0], item.split("_")[1]) in acoustic_dict.keys():
#             # pull out the acoustic feats dataframe
#             acoustic_data = acoustic_dict[(item.split("_")[0], item.split("_")[1])]
#
#             # add this dialogue + utt combo to the list of possible ones
#             usable_utts.append((item.split("_")[0], item.split("_")[1]))
#
#             if not avgd and not add_avging:
#                 # set size of acoustic data holder
#                 acoustic_holder = [[0] * acoustic_length] * longest_acoustic
#
#                 for i, row in enumerate(acoustic_data):
#                     # for now, using longest acoustic file in TRAIN only
#                     if i >= longest_acoustic:
#                         break
#                     # needed because some files allegedly had length 0
#                     for j, feat in enumerate(row):
#                         acoustic_holder[i][j] = feat
#             else:
#                 if avgd:
#                     acoustic_holder = acoustic_data
#                 elif add_avging:
#                     acoustic_holder = torch.mean(torch.tensor(acoustic_data), dim=0)
#
#             all_acoustic.append(torch.tensor(acoustic_holder))
#
#     return all_acoustic, usable_utts
