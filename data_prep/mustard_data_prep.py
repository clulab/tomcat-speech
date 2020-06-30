# prepare the data from MUStARD dataset

import os
import json
from data_prep.audio_extraction import convert_mp4_to_wav, ExtractAudio
import pandas as pd


def organize_labels_from_json(jsonfile, savepath, save_name):
    """
    Take the jsonfile containing the text,
    """
    with open(jsonfile, 'r') as jfile:
        json_data = json.load(jfile)

    # create holder for relevant information
    data_holder = [["clip_id", "utterance", "speaker", "sarcasm"]]

    # get utterance, speaker, clip ID, and gold labels
    for clip in json_data.keys():
        clip_id = clip
        utt = json_data[clip]["utterance"]
        spk = json_data[clip]["speaker"]
        sarc = str(1 if json_data[clip]["sarcasm"] else 0)

        data_holder.append([clip_id, utt, spk, sarc])

    # save the data to a new csv file
    with open(os.path.join(savepath, save_name), 'w') as savefile:
        for item in data_holder:
            savefile.write("\t".join(item))
            savefile.write("\n")


def create_data_folds(data, perc_train, perc_test):
    """
    Create train, dev, and test folds for a dataset without them
    Specify the percentage of the data that goes into each fold
    data : a Pandas dataframe with (at a minimum) gold labels for all data
    perc_* : the percentage for each fold
    Percentage not included in train or test fold allocated to dev
    """
    # shuffle the rows of the dataframe
    shuffled = data.sample(frac=1).reset_index(drop=True)

    # get length of df
    length = shuffled.shape[0]

    # calculate length of each split
    train_len = perc_train * length
    test_len = perc_test * length

    # get slices of dataset
    train_data = shuffled.iloc[:int(train_len)]
    test_data = shuffled.iloc[int(train_len):int(train_len) + int(test_len)]
    dev_data = shuffled.iloc[int(train_len) + int(test_len):]

    # return data
    return train_data, dev_data, test_data


def preprocess_mustard_data(base_path, gold_save_name, acoustic_save_dir,
                            smile_path, acoustic_feature_set="IS10"):
    """
    Preprocess the mustard data by getting an organized CSV from the json gold file,
    converting mp4 clips to wav files and extracting acoustic features from the wavs
    base_path : the path to the base MUStARD directory
    gold_save_name : the name of the tsv file to save gold labels + utterances to
    acoustic_save_dir : the directory in which to save acoustic feature files
    smile_path : the path to OpenSMILE
    acoustic_feature_set : the feature set to use with ExtractAudio
    """
    # set path to the json file (named by dataset authors)
    json_path = os.path.join(base_path, "sarcasm_data.json")

    # extract relevant info from json files and save as csv
    organize_labels_from_json(json_path, base_path, gold_save_name)

    # set path to mp4 files
    path_to_files = os.path.join(base_path, "utterances_final")

    # convert mp4 files to wav
    for clip in os.listdir(path_to_files):
        convert_mp4_to_wav(os.path.join(path_to_files, clip))

    # set path to acoustic feats
    acoustic_save_path = os.path.join(json_path, acoustic_save_dir)
    # create the save directory if it doesn't exist
    if not os.path.exists(acoustic_save_path):
        os.makedirs(acoustic_save_path)

    # extract features using opensmile
    for audio_file in os.listdir(path_to_files):
        if audio_file.endswith(".wav"):
            audio_name = audio_file.split(".wav")[0]
            audio_save_name = str(audio_name) + "_" + acoustic_feature_set + ".csv"
            extractor = ExtractAudio(path_to_files, audio_file, acoustic_save_path,
                                     smile_path)
            extractor.save_acoustic_csv(feature_set=acoustic_feature_set,
                                        savename=audio_save_name)
