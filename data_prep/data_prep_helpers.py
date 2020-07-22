# prepare text and audio for use in neural network models
import os
import sys
from collections import OrderedDict

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.feature_selection import SelectKBest, chi2

import statistics


# classes


class DatumListDataset(Dataset):
    """
    A dataset to hold a list of datums
    """

    def __init__(self, data_list, class_weights=None):
        self.data_list = data_list

        self.class_weights = class_weights

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        """
        item (int) : the index to a data point
        """
        return self.data_list[item]

    def targets(self):
        for datum in self.data_list:
            yield datum[4]


class Glove(object):
    def __init__(self, glove_dict):
        """
        Use a dict of format {word: vec} to get torch.tensor of vecs
        :param glove_dict: a dict created with make_glove_dict
        """
        self.glove_dict = OrderedDict(glove_dict)
        self.data = self.create_embedding()
        self.wd2idx = self.get_index_dict()
        self.idx2glove = self.get_index2glove_dict()  # todo: get rid of me
        self.max_idx = -1

        # add an average <UNK> if not in glove dict
        if "<UNK>" not in self.glove_dict.keys():
            mean_vec = self.get_avg_embedding()
            self.add_vector("<UNK>", mean_vec)

    def id_or_unk(self, t):
        if t.strip() in self.wd2idx:
            return self.wd2idx[t]
        else:
            # print(f"OOV: [[{t}]]")
            return self.wd2idx["<UNK>"]

    def index(self, toks):
        return [self.id_or_unk(t) for t in toks]

    def create_embedding(self):
        emb = []
        for vec in self.glove_dict.values():
            emb.append(vec)
        return torch.tensor(emb)

    def get_embedding_from_index(self, idx):
        return self.idx2glove[idx]

    def get_index_dict(self):
        # create word: index dict
        c = 0
        wd2idx = {}
        for k in self.glove_dict.keys():
            wd2idx[k] = c
            c += 1
        self.max_idx = c
        return wd2idx

    def get_index2glove_dict(self):
        # create index: vector dict
        c = 0
        idx2glove = {}
        for k, v in self.glove_dict.items():
            idx2glove[self.wd2idx[k]] = v
        return idx2glove

    def add_vector(self, word, vec):
        # adds a new word vector to the dictionaries
        self.max_idx += 1
        if self.max_idx not in self.wd2idx.keys():
            self.glove_dict[word] = vec  # add to the glove dict
            self.wd2idx[word] = self.max_idx  # add to the wd2idx dict
            self.idx2glove[self.max_idx] = vec  # add to the idx2 glove dict
            torch.cat(
                (self.data, vec.unsqueeze(dim=0)), dim=0
            )  # add to the data tensor

    def get_avg_embedding(self):
        # get an average of all embeddings in dataset
        # can be used for "<UNK>" if it doesn't exist
        return torch.mean(self.data, dim=0)


class MinMaxScaleRange:
    """
    A class to calculate mins and maxes for each feature in the data in order to
    use min-max scaling
    """

    def __init__(self,):
        self.mins = {}
        self.maxes = {}

    def update(self, key, val):
        if (
            key in self.mins.keys() and val < self.mins[key]
        ) or key not in self.mins.keys():
            self.mins[key] = val
        if (
            key in self.maxes.keys() and val > self.maxes[key]
        ) or key not in self.maxes.keys():
            self.maxes[key] = val

    def contains(self, key):
        if key in self.mins.keys():
            return True
        else:
            return False

    def min(self, key):
        try:
            return self.mins[key]
        except KeyError:
            return "The key {0} does not exist in mins".format(key)

    def max(self, key):
        try:
            return self.maxes[key]
        except KeyError:
            return "The key {0} does not exist in maxes".format(key)


# helper functions


def clean_up_word(word):
    word = word.replace("\x92", "'")
    word = word.replace("\x91", "")
    word = word.replace("\x97", "-")
    # clean up word by putting in lowercase + removing punct
    punct = [
        ",",
        ".",
        "!",
        "?",
        ";",
        ":",
        "'",
        '"',
        "-",
        "$",
        "’",
        "…",
        "[",
        "]",
        "(",
        ")",
    ]
    for char in word:
        if char in punct:
            word = word.replace(char, " ")
    if word.strip() == "":
        word = "<UNK>"
    return word


def get_avg_vec(nested_list):
    # get the average vector of a nested list
    # used for utterance-level feature averaging
    return [statistics.mean(item) for item in zip(*nested_list)]


def get_class_weights(y_set):
    class_counts = {}
    y_values = y_set.tolist()

    num_labels = max(y_values) + 1

    for item in y_values:
        if item not in class_counts:
            class_counts[item] = 1
        else:
            class_counts[item] += 1
    class_weights = [0.0] * num_labels
    for k, v in class_counts.items():
        class_weights[k] = float(v)
    class_weights = torch.tensor(class_weights)
    return class_weights


def get_gender_avgs(acoustic_set, gender_set, gender=1):
    """
    Get averages and standard deviations split by gender
    param acoustic_set : the acoustic data
    param gender : the gender to return avgs for; 0 = all, 1 = f, 2 = m
    """
    all_items = []

    for i, item in enumerate(acoustic_set):
        if gender_set[i] == gender:
            all_items.append(torch.tensor(item))

    all_items = torch.stack(all_items)

    mean = all_items.mean(dim=0, keepdim=False)
    stdev = all_items.std(dim=0, keepdim=False)

    return mean, stdev


def get_longest_utterance(pd_dataframes):
    """
    Get the longest utterance in the dataset
    :param pd_dataframes: the dataframes for the dataset
    :return:
    """
    max_length = 0
    for item in pd_dataframes:
        for i in range(item["utt_num"].max()):
            utterance = item.loc[item["utt_num"] == i + 1]
            utt_length = utterance.shape[0]
            if utt_length > max_length:
                max_length = utt_length
    return max_length


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


def get_speaker_gender(idx2gender_path):
    """
    Get the gender of each speaker in the list
    Includes 0 as UNK, 1 == F, 2 == M
    """
    speaker_df = pd.read_csv(idx2gender_path, usecols=["idx", "gender"])

    return dict(zip(speaker_df.idx, speaker_df.gender))


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


def make_acoustic_dict(
    acoustic_path,
    f_end="_IS09_avgd.csv",
    use_cols=None,
    data_type="clinical",
    files_to_get=None,
):
    """
    makes a dict of (sid, call): data for use in ClinicalDataset objects
    f_end: end of acoustic file names
    use_cols: if set, should be a list [] of column names to include
    """
    acoustic_dict = {}
    for f in os.listdir(acoustic_path):
        if f.endswith(f_end):
            if files_to_get is None or "_".join(f.split("_")[:2]) in files_to_get:
                if use_cols is not None:
                    feats = pd.read_csv(acoustic_path + "/" + f, usecols=use_cols)
                else:
                    feats = pd.read_csv(acoustic_path + "/" + f)
                sid = f.split("_")[0]
                if data_type == "asist":
                    callid = f.split("_")[2]  # asist data has format sid_mission_num
                else:
                    # clinical data has format sid_callid
                    # meld has format dia_utt
                    callid = f.split("_")[1]
                acoustic_dict[(sid, callid)] = feats
    return acoustic_dict


def make_acoustic_set(
    text_path,
    acoustic_dict,
    data_type,
    acoustic_length,
    longest_acoustic,
    add_avging=True,
    avgd=False,
):
    """
    Prep the acoustic data using the acoustic dict
    :param text_path: FULL path to file containing utterances + labels
    :param acoustic_dict:
    :param add_avging: whether to average the feature sets
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
    if data_type == "meld":
        valid_dia_utt = all_utts_df["DiaID_UttID"].tolist()
    else:
        valid_dia_utt = all_utts_df["clip_id"].tolist()

    # set holders for acoustic data
    all_acoustic = []
    usable_utts = []

    # for all items with audio + gold label
    for idx, item in enumerate(valid_dia_utt):
        # if that dialogue and utterance appears has an acoustic feats file
        if (item.split("_")[0], item.split("_")[1]) in acoustic_dict.keys():

            # pull out the acoustic feats dataframe
            acoustic_data = acoustic_dict[(item.split("_")[0], item.split("_")[1])]

            # add this dialogue + utt combo to the list of possible ones
            usable_utts.append((item.split("_")[0], item.split("_")[1]))

            if not avgd and not add_avging:
                # set intermediate acoustic holder
                acoustic_holder = [[0] * acoustic_length] * longest_acoustic

                # add the acoustic features to the holder of features
                for i, feats in enumerate(acoustic_data):
                    # for now, using longest acoustic file in TRAIN only
                    if i >= longest_acoustic:
                        break
                    # needed because some files allegedly had length 0
                    for j, feat in enumerate(feats):
                        acoustic_holder[i][j] = feat
            else:
                if avgd:
                    acoustic_holder = acoustic_data
                elif add_avging:
                    acoustic_holder = torch.mean(torch.tensor(acoustic_data), dim=0)

            # add features as tensor to acoustic data
            all_acoustic.append(torch.tensor(acoustic_holder))

    # pad the sequence and reshape it to proper format
    all_acoustic = nn.utils.rnn.pad_sequence(all_acoustic)
    all_acoustic = all_acoustic.transpose(0, 1)

    return all_acoustic, usable_utts


def make_glove_dict(glove_path):
    """creates a dict of word: embedding pairs
    :param glove_path: the path to our glove file
    (includes name of file and extension)
    """
    glove_dict = {}
    with open(glove_path) as glove_file:
        for line in glove_file:
            line = line.rstrip().split(" ")
            glove_dict[line[0]] = [float(item) for item in line[1:]]
    return glove_dict


def perform_feature_selection(xs, ys, num_to_keep):
    """
    Perform feature selection on the dataset
    """
    new_xs = SelectKBest(chi2, k=num_to_keep).fit_transform(xs, ys)
    return new_xs


def scale_feature(value, min_val, max_val, lower=0.0, upper=1.0):
    # scales a single feature using min-max normalization
    if min_val == max_val:
        return upper
    else:
        # the result will be a value in [lower, upper]
        return lower + (upper - lower) * (value - min_val) / (max_val - min_val)


def transform_acoustic_item(item, acoustic_means, acoustic_stdev):
    """
    Use gender averages and stdev to transform an acoustic item
    item : a 1D tensor
    acoustic_means : the appropriate vector of means
    acoustic_stdev : the corresponding stdev vector
    """
    return (item - acoustic_means) / acoustic_stdev
