from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from collections import OrderedDict
import pandas as pd
import numpy as np
import functools
import operator
import os

from data_prep.data_prep_helpers import (
    MinMaxScaleRange,
    get_longest_utterance,
    clean_up_word,
    get_avg_vec,
    scale_feature,
)


class ClinicalDataset(Dataset):
    def __init__(
        self,
        acoustic_dict,
        glove,
        ys_path,
        splits=5,
        cols_to_skip=3,
        norm="minmax",
        sequence_prep=None,
        truncate_from="start",
        alignment=None,
    ):
        """
        :param acoustic_dict: dict of {(sid, call) : data}
        :param glove: an instance of class Glove
        :param ys_path: path to dataframe of sid + ys
        :param splits: number of splits for CV
        :param norm: the type of data normalization
        :param sequence_prep: the way sequences are handled, options: truncate, pad, None
        :param truncate_from: whether to truncate from start or end of file
        """
        self.cols_to_skip = cols_to_skip
        self.acoustic_dict = OrderedDict(acoustic_dict)
        self.glove = glove
        self.ys_df = pd.read_csv(ys_path)
        self.norm = norm
        self.sequence_prep = sequence_prep
        self.truncate_from = truncate_from
        self.alignment = alignment

        if norm == "minmax":
            # currently uses index-keyed min-max for features
            # should we change it later?
            self.min_max_scaler = MinMaxScaleRange()
            self.get_min_max_scales()

        self.valid_files = self.ys_df["sid"].tolist()
        self.skipped_files = []

        if self.alignment == "utt":
            (
                self.x_acoustic,
                self.x_glove,
                self.x_speaker,
            ) = self.combine_data_utt_level()
        else:
            (
                self.x_acoustic,
                self.x_glove,
                self.x_speaker,
            ) = self.combine_acoustic_and_glove()
        self.y_data = self.create_ordered_ys()
        self.data = self.combine_xs_and_ys()
        self.splits = splits
        self.data_for_model_input = self.get_data_splits()

        # for working with an individual split
        self.current_split = []
        self.current_split_num = 0
        self.val_split = []
        self.remaining_splits = []

        self.set_split(0)

    def get_min_max_scales(self):
        for call in self.acoustic_dict.values():
            for row in call.itertuples():
                for i, wd in enumerate(row):
                    if i >= self.cols_to_skip + 1:
                        self.min_max_scaler.update((i - (self.cols_to_skip + 1)), wd)

    def __len__(self):
        return len(self.current_split)

    def __getitem__(self, index):
        return self.current_split[index]

    def set_split(self, n):
        # set the split; n is the TEST split, n-1 is DEV split
        self.current_split = self.data_for_model_input[n]
        self.current_split_num = n
        if n - 1 >= 0:
            prev = n - 1
        else:
            prev = max(self.data_for_model_input.keys())
        self.val_split = self.data_for_model_input[prev]
        remaining_splits = [
            val
            for key, val in self.data_for_model_input.items()
            if (key != n and key != prev)
        ]
        self.remaining_splits = functools.reduce(operator.iconcat, remaining_splits, [])

    def get_data_splits(self):
        data_dict = {}
        # calculate length of data
        total_len = len(self.data)
        # calculate length of each split based on number
        split_len = int(total_len / self.splits)
        # random permutation of indices
        indices = np.random.permutation(total_len)
        # get split indices
        c = 0
        # for all but final split
        for split in range(self.splits - 1):
            split_data = []
            split_idx = indices[split_len * c : split_len * (c + 1)]
            # get data for each of the indices
            for i in split_idx:
                split_data.append(self.data[i])
            # add to dict of splits
            data_dict[c] = split_data
            c += 1
        # for final split do the same
        split_idx = indices[split_len * c :]
        split_data = []
        for i in split_idx:
            split_data.append(self.data[i])
        data_dict[c] = split_data
        return data_dict

    def combine_data_utt_level(self):
        """
        Combine acoustic feats, glove indices, and speaker info
        Used when data needs to be aligned at the utt level
        Separate from the below because fixme
        :return:
        """
        print("Utt level alignment and normalization starting")

        # set holders for acoustic, words, and speakers
        acoustic_data = []
        ordered_words = []
        ordered_speakers = []

        # skip the the first cols_to_skip columns
        start_idx = self.cols_to_skip

        # counter for smallest dataframe for truncation
        # get skipped files based on number of items
        #   e.g. too few utts
        if self.sequence_prep == "truncate":
            smallest = self.truncate_seq()

        # get longest utterance
        longest_utt = get_longest_utterance(self.acoustic_dict.values())

        # iterate through items in the acoustic dict
        for key, item in self.acoustic_dict.items():

            # if the item has gold data
            if key[0] in self.valid_files and key[0] not in self.skipped_files:

                # prepare intermediate holders
                intermediate_wds = []
                intermediate_speakers = []
                intermediate_acoustic = []

                for i in range(item["utt_num"].max()):

                    utterance = item.loc[item["utt_num"] == i + 1]

                    utt_wds = [0] * longest_utt
                    # utt_speakers = []
                    utt_acoustic = []

                    utt_speaker = utterance[
                        "speaker"
                    ].max()  # they should all be the same number

                    wd_idx = 0

                    # for each row in that item's dataframe
                    # note: this idx is a row index out of the total in a df
                    #   it does NOT restart for each item
                    for idx, row in utterance.iterrows():

                        # get the word
                        wd = row["word"]
                        wd = clean_up_word(wd)
                        # save that word's index
                        if wd in self.glove.wd2idx.keys():
                            utt_wds[wd_idx] = self.glove.wd2idx[wd]
                            # utt_wds.append(self.glove.wd2idx[wd])
                        else:
                            utt_wds[wd_idx] = self.glove.wd2idx["<UNK>"]
                            # utt_wds.append(self.glove.wd2idx["<UNK>"])

                        # save the acoustic information in remaining columns
                        row_vals = row.values[start_idx:].tolist()

                        # if using min-max scaling, scale the data
                        if self.norm == "minmax":
                            self.minmax_scale(row_vals, lower=0, upper=1)

                        # add acoustic information to intermediate holder
                        utt_acoustic.append(row_vals)

                        wd_idx += 1

                    utt_avg_acoustic = get_avg_vec(utt_acoustic)
                    intermediate_acoustic.append(utt_avg_acoustic)
                    intermediate_speakers.append([utt_speaker])
                    intermediate_wds.append(utt_wds)

                # add information from intermediate holder to lists of all data
                if self.sequence_prep == "truncate":
                    acoustic_data.append(intermediate_acoustic)
                    ordered_words.append(intermediate_wds)
                    ordered_speakers.append(intermediate_speakers)
                else:
                    acoustic_data.append(torch.tensor(intermediate_acoustic))
                    ordered_words.append(torch.tensor(intermediate_wds))
                    ordered_speakers.append(torch.tensor(intermediate_speakers))

        # use zero-padding to make all sequences the same length
        # if we need to pad, we MUST pack
        if self.sequence_prep == "pad":
            acoustic_data = nn.utils.rnn.pad_sequence(acoustic_data)
            ordered_words = nn.utils.rnn.pad_sequence(ordered_words)
            ordered_speakers = nn.utils.rnn.pad_sequence(ordered_speakers)

            # swap axes to get (total_inputs, length_of_sequence, length_of_vector)
            acoustic_data = acoustic_data.transpose(0, 1)
            ordered_words = ordered_words.transpose(0, 1)
            ordered_speakers = ordered_speakers.transpose(0, 1)

        elif self.sequence_prep == "truncate":
            if self.truncate_from == "start":
                acoustic_data = [item[-smallest:] for item in acoustic_data]
                ordered_words = [item[-smallest:] for item in ordered_words]
                ordered_speakers = [item[-smallest:] for item in ordered_speakers]
            else:
                acoustic_data = [item[:smallest] for item in acoustic_data]
                ordered_words = [item[:smallest] for item in ordered_words]
                ordered_speakers = [item[:smallest] for item in ordered_speakers]

            acoustic_data = torch.tensor(acoustic_data)
            ordered_words = torch.tensor(ordered_words)
            ordered_speakers = torch.tensor(ordered_speakers)

        print("Size of acoustic data is now: " + str(acoustic_data.shape))

        print("Data prep and normalization complete")

        # return acoustic info, words indices, speaker
        return acoustic_data, ordered_words, ordered_speakers

    def combine_acoustic_and_glove(self):
        """
        Combine acoustic feats + glove indices (speaker info, too)
        :return: list of double of ([acoustic], [glove_index], [speaker])
        """
        print("Data prep and normalization starting")
        # print(self.acoustic_dict.keys())
        # print(self.valid_files)
        # sys.exit()

        # set holders for acoustic, words, and speakers
        acoustic_data = []
        ordered_words = []
        ordered_speakers = []

        # skip the the first cols_to_skip columns
        start_idx = self.cols_to_skip

        # counter for smallest dataframe for truncation
        # get skipped files based on number of items
        #   e.g. too few utts
        if self.sequence_prep == "truncate":
            smallest = self.truncate_seq()

        # iterate through items in the acoustic dict
        for key, item in self.acoustic_dict.items():
            # if the item has gold data
            if key[0] in self.valid_files:
                # prepare intermediate holders
                intermediate_wds = []
                intermediate_speakers = []
                intermediate_acoustic = []

                # for each row in that item's dataframe
                for idx, row in item.iterrows():
                    # get the speaker
                    spkr = row["speaker"]
                    print(spkr)
                    intermediate_speakers.append(
                        spkr - 1
                    )  # speakers are currently 1 and 2, we want 0 and 1

                    # get the word
                    wd = row["word"]
                    wd = clean_up_word(wd)
                    # save that word's index
                    if wd in self.glove.wd2idx.keys():
                        intermediate_wds.append(self.glove.wd2idx[wd])
                    else:
                        intermediate_wds.append(self.glove.wd2idx["<UNK>"])

                    # save the acoustic information in remaining columns
                    row_vals = row.values[start_idx:].tolist()

                    # if using min-max scaling, scale the data
                    if self.norm == "minmax":
                        self.minmax_scale(row_vals, lower=0, upper=1)
                    # add acoustic information to intermediate holder
                    intermediate_acoustic.append(row_vals)

                # add information from intermediate holder to lists of all data
                if self.sequence_prep == "truncate":
                    acoustic_data.append(intermediate_acoustic)
                    ordered_words.append(intermediate_wds)
                    ordered_speakers.append(intermediate_speakers)
                else:
                    acoustic_data.append(torch.tensor(intermediate_acoustic))
                    ordered_words.append(torch.tensor(intermediate_wds))
                    ordered_speakers.append(torch.tensor(intermediate_speakers))

        # use zero-padding to make all sequences the same length
        # if we need to pad, we MUST pack
        if self.sequence_prep == "pad":
            acoustic_data = nn.utils.rnn.pad_sequence(acoustic_data)
            ordered_words = nn.utils.rnn.pad_sequence(ordered_words)
            ordered_speakers = nn.utils.rnn.pad_sequence(ordered_speakers)

            # swap axes to get (total_inputs, length_of_sequence, length_of_vector)
            acoustic_data = acoustic_data.transpose(0, 1)
            ordered_words = ordered_words.transpose(0, 1)
            ordered_speakers = ordered_speakers.transpose(0, 1)

        elif self.sequence_prep == "truncate":
            if self.truncate_from == "start":
                acoustic_data = [item[-smallest:] for item in acoustic_data]
                ordered_words = [item[-smallest:] for item in ordered_words]
                ordered_speakers = [item[-smallest:] for item in ordered_speakers]
            else:
                acoustic_data = [item[:smallest] for item in acoustic_data]
                ordered_words = [item[:smallest] for item in ordered_words]
                ordered_speakers = [item[:smallest] for item in ordered_speakers]

            acoustic_data = torch.tensor(acoustic_data)
            ordered_words = torch.tensor(ordered_words)
            ordered_speakers = torch.tensor(ordered_speakers)
        print("Acoustic data size is: " + str(acoustic_data.shape))
        print("Ordered words is: " + str(ordered_words.shape))

        print("Data prep and normalization complete")

        # return acoustic info, words indices, speaker
        return acoustic_data, ordered_words, ordered_speakers

    def create_ordered_ys(self):
        """
        create a list of all outcomes in the same order as the data
        """
        # create holder
        ordered_ys = []

        # set index for ys dataframe to sid to use it in search
        ys = self.ys_df.set_index(["sid"])

        # for each (sid, callid) pair in acoustic dict's keys
        for tup in self.acoustic_dict.keys():
            # if the sid has gold data, add it
            if tup[0] in self.valid_files:
                print(tup)
                if tup not in self.skipped_files:
                    y = ys[ys.index == tup[0]].overall.item()
                    ordered_ys.append(y)

        # return ordered list
        return ordered_ys

    def combine_xs_and_ys(self):
        # combine all x and y data into list of tuples for easier access with DataLoader
        all_data = []

        for i, item in enumerate(self.x_acoustic):
            # print(i)
            all_data.append((item, self.x_glove[i], self.x_speaker[i], self.y_data[i]))

        return all_data

    def minmax_scale(self, df, lower=0, upper=1):
        # perform min-max scaling
        for i, val in enumerate(df):
            new_val = scale_feature(
                val,
                self.min_max_scaler.min(i),
                self.min_max_scaler.max(i),
                lower,
                upper,
            )
            df[i] = new_val
        return df

    def truncate_seq(self, starting_size=1e10, minimum=1000):
        """
        truncate the sequence of data
        :return smallest_item: the size of the smallest item
        """
        smallest_item = starting_size
        skipped_files = []
        for key, item in self.acoustic_dict.items():
            size = item.shape[0]
            if size < smallest_item:
                if size >= minimum:
                    smallest_item = size
                else:
                    skipped_files.append(key)
        self.skipped_files = skipped_files

        return smallest_item
