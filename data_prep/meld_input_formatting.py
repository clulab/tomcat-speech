# prepare MELD input for usage in networks

import os
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset

from data_prep.data_prep import clean_up_word
from collections import OrderedDict

import torchtext
from torchtext.data import get_tokenizer
from torch.nn.utils.rnn import pad_sequence


class MELDData(Dataset):
    """
    A dataset to manipulate the MELD data before passing to NNs
    """
    def __init__(self, meld_path, glove, acoustic_length, f_end="_IS10.csv", use_cols=None, avgd=True,
                 add_avging=True):
        self.train_path = meld_path + "/train"
        self.dev_path = meld_path + "/dev"
        self.test_path = meld_path + "/test"
        self.meld_train = "{0}/train_sent_emo.csv".format(self.train_path)
        self.meld_dev = "{0}/dev_sent_emo.csv".format(self.dev_path)
        self.meld_test = "{0}/test_sent_emo.csv".format(self.test_path)
        self.tokenizer = get_tokenizer("basic_english")

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

        # get the number of acoustic features
        self.acoustic_length = acoustic_length

        # Glove object
        self.glove = glove

        # get speaker to gender dict
        self.speaker2gender = get_speaker_gender(f"{meld_path}/speaker2idx.csv")

        print("Collecting acoustic features")

        # ordered dicts of acoustic data
        self.train_dict, self.train_acoustic_lengths = make_acoustic_dict_meld("{0}/{1}".format(self.train_path,
                                                                                                self.train_dir),
                                                                               f_end, use_cols, avgd=avgd)
        self.train_dict = OrderedDict(self.train_dict)
        self.dev_dict, self.dev_acoustic_lengths = make_acoustic_dict_meld("{0}/{1}".format(self.dev_path,
                                                                                            self.dev_dir),
                                                                           f_end, use_cols, avgd=avgd)
        self.dev_dict = OrderedDict(self.dev_dict)
        self.test_dict, self.test_acoustic_lengths = make_acoustic_dict_meld("{0}/{1}".format(self.test_path,
                                                                                              self.test_dir),
                                                                             f_end, use_cols, avgd=avgd)
        self.test_dict = OrderedDict(self.test_dict)

        # utterance-level dict
        self.longest_utt, self.longest_dia = self.get_longest_utt_meld()

        print("Finalizing acoustic organization")

        self.train_acoustic, self.train_usable_utts = self.make_acoustic_set(self.meld_train, self.train_dict,
                                                                             self.train_acoustic_lengths,
                                                                             add_avging=add_avging)
        self.dev_acoustic, self.dev_usable_utts = self.make_acoustic_set(self.meld_dev, self.dev_dict,
                                                                         self.train_acoustic_lengths,
                                                                         add_avging=add_avging) # keep train
        self.test_acoustic, self.test_usable_utts = self.make_acoustic_set(self.meld_test, self.test_dict,
                                                                           self.train_acoustic_lengths,
                                                                           add_avging-add_avging) # keep train

        print("Getting text, speaker, and y features")

        # get utterance, speaker, y matrices for train, dev, and test sets
        self.train_utts, self.train_spkrs, self.train_genders, \
            self.train_y_emo, self.train_y_sent, self.train_utt_lengths = \
            self.make_meld_data_tensors(self.meld_train, self.train_usable_utts)

        self.dev_utts, self.dev_spkrs, self.dev_genders,\
            self.dev_y_emo, self.dev_y_sent, self.dev_utt_lengths = \
            self.make_meld_data_tensors(self.meld_dev, self.dev_usable_utts)

        self.test_utts, self.test_spkrs, self.test_genders,\
            self.test_y_emo, self.test_y_sent, self.test_utt_lengths = \
            self.make_meld_data_tensors(self.meld_test, self.test_usable_utts)

        # acoustic feature normalization based on train
        self.all_acoustic_means = torch.stack(self.train_acoustic).mean(dim=0, keepdim=False)
        self.all_acoustic_deviations = torch.stack(self.train_acoustic).std(dim=0, keepdim=False)

        self.male_acoustic_means, self.male_deviations = self.get_gender_avgs(gender=2)
        self.female_acoustic_means, self.female_deviations = self.get_gender_avgs(gender=1)

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
            # normalize
            item_transformed = item
            if self.train_genders[i] == 1:
                item_transformed = self.transform_acoustic_item(item, self.train_genders[i])
            # item_transformed = (item - self.acoustic_means) / self.acoustic_deviations
            train_data.append((item_transformed, self.train_utts[i], self.train_spkrs[i], self.train_genders[i],
                               self.train_y_emo[i], self.train_y_sent[i], self.train_utt_lengths[i],
                               self.train_acoustic_lengths[i]))

        for i, item in enumerate(self.dev_acoustic):
            item_transformed = self.transform_acoustic_item(item, self.dev_genders[i])
            # item_transformed = (item - self.acoustic_means) / self.acoustic_deviations
            dev_data.append((item_transformed, self.dev_utts[i], self.dev_spkrs[i], self.dev_genders[i],
                             self.dev_y_emo[i], self.dev_y_sent[i], self.dev_utt_lengths[i],
                             self.dev_acoustic_lengths[i]))

        for i, item in enumerate(self.test_acoustic):
            item_transformed = self.transform_acoustic_item(item, self.test_genders[i])
            # item_transformed = (item - self.acoustic_means) / self.acoustic_deviations
            test_data.append((item_transformed, self.test_utts[i], self.test_spkrs[i], self.test_genders[i],
                              self.test_y_emo[i], self.test_y_sent[i], self.test_utt_lengths[i],
                              self.test_acoustic_lengths[i]))

        return train_data, dev_data, test_data

    def get_gender_avgs(self, gender=1):
        """
        Get averages and standard deviations split by gender
        param gender : the gender to return avgs for; 0 = all, 1 = f, 2 = m
        """
        all_items = []

        for i, item in enumerate(self.train_acoustic):
            if self.train_genders[i] == gender:
                all_items.append(torch.tensor(item))

        print(all_items[0].shape)
        print(len(all_items))

        all_items = torch.stack(all_items)

        mean = all_items.mean(dim=0, keepdim=False)
        stdev = all_items.std(dim=0, keepdim=False)

        print(mean)
        print(stdev)

        return mean, stdev

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

        for i, item in enumerate(all_utts):
            item = clean_up_word(item)
            item = self.tokenizer(item)
            if len(item) > longest:
                longest = len(item)

        # get longest dialogue length
        longest_dia = max(all_utts_df['Utterance_ID'].tolist()) + 1  # because 0-indexed

        print(longest)

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
        all_genders = []
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
                utts = [0] * self.longest_utt

                # get values from row
                utt = clean_up_word(row["Utterance"])
                utt = self.tokenizer(utt)
                utt_lengths.append(len(utt))

                spk_id = row['Speaker']
                gen = self.speaker2gender[spk_id]
                emo = row['Emotion']
                sent = row['Sentiment']

                # convert words to indices for glove
                utt_indexed = self.glove.index(utt)
                for i, item in enumerate(utt_indexed):
                    if i >= self.longest_utt:
                        print(i)
                    utts[i] = item

                all_utts.append(torch.tensor(utts))
                # all_utts.append(torch.tensor(utt_indexed))
                all_speakers.append([spk_id])
                all_genders.append(gen)
                all_emotions.append(emo)
                all_sentiments.append(sent)

        # create pytorch tensors for each
        all_speakers = torch.tensor(all_speakers)
        all_genders = torch.tensor(all_genders)
        all_emotions = torch.tensor(all_emotions)
        all_sentiments = torch.tensor(all_sentiments)
        all_utts = pad_sequence(all_utts, batch_first=True, padding_value=0)

        # return data
        return all_utts, all_speakers, all_genders, all_emotions, all_sentiments, utt_lengths

    def make_acoustic_set(self, text_path, acoustic_dict, acoustic_lengths, add_avging=True):
        """
        Prep the acoustic data using the acoustic dict
        :param text_path: FULL path to file containing utterances + labels
        :param acoustic_dict:
        :param acoustic_lengths:
        :param add_avging: whether to average the feature sets
        :return:
        """
        # get longest acoustic df
        longest_acoustic = max(acoustic_lengths)

        # read in the acoustic csv
        all_utts_df = pd.read_csv(text_path)
        # get lists of valid dialogues and utterances
        valid_dia_utt = all_utts_df['DiaID_UttID'].tolist()

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

                if not self.avgd and not add_avging:
                    # set size of acoustic data holder
                    acoustic_holder = [[0] * self.acoustic_length] * longest_acoustic

                    for i, row in enumerate(acoustic_data):
                        # print("i'm in the right spot")
                        # for now, using longest acoustic file in TRAIN only
                        if i >= longest_acoustic:
                            break
                        # needed because some files allegedly had length 0
                        for j, feat in enumerate(row):
                            acoustic_holder[i][j] = feat
                else:
                    # print("something is wrong...")
                    if self.avgd:
                        # print("self.avgd is true")
                        acoustic_holder = acoustic_data
                    elif add_avging:
                        # print("add_avging is true")
                        acoustic_holder = torch.mean(torch.tensor(acoustic_data), dim=0)

                all_acoustic.append(torch.tensor(acoustic_holder))

        # print(all_acoustic[0].shape)
        # print(len(all_acoustic))
        # sys.exit()

        return all_acoustic, usable_utts

    def transform_acoustic_item(self, item, gender):
        """
        Use gender averages and stdev to transform an acoustic item
        item : a 1D tensor
        gender : an int (1=female, 2=male, 0=all)
        """
        if gender == 1:
            item_transformed = (item - self.female_acoustic_means) / self.female_deviations
        elif gender == 2:
            item_transformed = (item - self.male_acoustic_means) / self.male_deviations
        else:
            item_transformed = (item - self.all_acoustic_means) / self.all_acoustic_deviations

        return item_transformed


# helper functions
def get_class_weights(y_set):
    class_counts = {}
    y_values = y_set.tolist()

    num_labels = max(y_values) + 1
    # y_values = [item.index(max(item)) for item in y_set.tolist()]

    for item in y_values:
        if item not in class_counts:
            class_counts[item] = 1
        else:
            class_counts[item] += 1
    class_weights = [0.0] * num_labels
    for k,v in class_counts.items():
        class_weights[k] = float(v)
    class_weights = torch.tensor(class_weights)
    return class_weights


def make_acoustic_dict_meld(acoustic_path, f_end="_IS10.csv", use_cols=None, avgd=True):
    """
    makes a dict of (sid, call): data for use in ClinicalDataset objects
    f_end: end of acoustic file names
    use_cols: if set, should be a list [] of column names to include
    """
    acoustic_dict = {}
    acoustic_lengths = []
    # find acoustic features files
    for f in os.listdir(acoustic_path):
        if f.endswith(f_end):
            # set the separator--averaged files are actually CSV, others are ;SV
            if avgd:
                separator = ","
            else:
                separator = ";"

            # read in the file as a dataframe
            if use_cols is not None:
                feats = pd.read_csv(acoustic_path + "/" + f, usecols=use_cols,
                                    sep=separator)
            else:
                feats = pd.read_csv(acoustic_path + "/" + f, sep=separator)
                if not avgd:
                    feats.drop(['name', 'frameTime'], axis=1, inplace=True)

            # get the dialogue and utterance IDs
            dia_id = f.split("_")[0]
            utt_id = f.split("_")[1]

            # save the dataframe to a dict with (dialogue, utt) as key
            if feats.shape[0] > 0:
                acoustic_dict[(dia_id, utt_id)] = feats.values.tolist()
                acoustic_lengths.append(feats.shape[0])

    return acoustic_dict, acoustic_lengths


def get_speaker_gender(idx2gender_path):
    """
    Get the gender of each speaker in the list
    Includes 0 as UNK, 1 == F, 2 == M
    """
    speaker_df = pd.read_csv(idx2gender_path, usecols=['idx', 'gender'])

    return dict(zip(speaker_df.idx, speaker_df.gender))
    # idxs = speaker_df['idx'].tolist()
    # gender = speaker_df['gender'].tolist()


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
