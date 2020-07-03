# prepare text and audio for use in neural network models
from collections import OrderedDict

import torch
from sklearn.feature_selection import SelectKBest, chi2
from torch.utils.data import Dataset

import statistics


def clean_up_word(word):
    # clean up word by putting in lowercase + removing punct
    punct = [",", ".", "!", "?", ";", ":"]  # don't include hyphen
    for char in word:
        if char in punct:
            word = word.replace(char, "")
    return word.lower()


def get_avg_vec(nested_list):
    # get the average vector of a nested list
    # used for utterance-level feature averaging
    return [statistics.mean(item) for item in zip(*nested_list)]


def make_glove_dict(glove_path):
    """creates a dict of word: embedding pairs
    :param glove_path: the path to our glove file
    (includes name of file and extension)
    """
    glove_dict = {}
    with open(glove_path) as glove_file:
        for line in glove_file:
            line = line.strip().split()
            glove_dict[line[0]] = [float(item) for item in line[1:]]
    return glove_dict


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
            torch.cat((self.data, vec.unsqueeze(dim=0)), dim=0)  # add to the data tensor

    def get_avg_embedding(self):
        # get an average of all embeddings in dataset
        # can be used for "<UNK>" if it doesn't exist
        return torch.mean(self.data, dim=0)


class MinMaxScaleRange:
    """
    A class to calculate mins and maxes for each feature in the data in order to
    use min-max scaling
    """
    def __init__(self, ):
        self.mins = {}
        self.maxes = {}

    def update(self, key, val):
        if (key in self.mins.keys() and val < self.mins[key]) or key not in self.mins.keys():
            self.mins[key] = val
        if (key in self.maxes.keys() and val > self.maxes[key]) or key not in self.maxes.keys():
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


def scale_feature(value, min_val, max_val, lower=0., upper=1.):
    # scales a single feature using min-max normalization
    if min_val == max_val:
        return upper
    else:
        # the result will be a value in [lower, upper]
        return lower + (upper - lower) * (value - min_val) / (max_val - min_val)


def get_longest_utterance(pd_dataframes):
    """
    Get the longest utterance in the dataset
    :param pd_dataframes: the dataframes for the dataset
    :return:
    """
    max_length = 0
    for item in pd_dataframes:
        for i in range(item['utt_num'].max()):
            utterance = item.loc[item['utt_num'] == i + 1]
            utt_length = utterance.shape[0]
            if utt_length > max_length:
                max_length = utt_length
                # print(max_length)
    return max_length


def feature_selection(xs, ys, num_to_keep):
    """
    Perform feature selection on the dataset
    """
    new_xs = SelectKBest(chi2, k=num_to_keep).fit_transform(xs, ys)
    return new_xs