"""
this file organizes prepared kaldi transcription files for the different datasets
each kaldi file is assumed to be in a "dataset_transcription.txt" file of the format
<filename.wav>\tCAPITALIZED_TRANSCRIPTION\n
does not include headers

"""

import os
import pandas as pd
import warnings
import sys

# read in file
# read in


class OrganizeKaldiTranscriptions:
    """
    Organizes kaldi transcriptions of datasets
    currently written for MELD, MUStARD, and ChaLearn
    dataset : a string of name of dataset
    location : full path to dataset base directory
    extension : the extension needed to access the transcribed file
    transcribed_file : the name of the transcribed file
    """
    def __init__(self, dataset, dataset_location, extension, transcribed_file):
        self.dataset = dataset.lower()  # options: 'meld', 'mustard', 'chalearn'
        self.location = dataset_location
        self.extension = extension
        # get list of extensions
        self.transcribed_file = transcribed_file
        self.transcribed_file_path = os.path.join(dataset_location, extension, transcribed_file)

    def read_in_current_files(self, current_file_location):
        """
        read in files containing gold transcriptions
        files are in different formats depending upon the dataset
        """
        # create dict for label : utt/other-info pairs
        data_dict = {}
        if self.dataset == "meld":
            # all utterances in a single csv file
            all_utts = pd.read_csv(current_file_location)
        elif self.dataset == "mustard" or self.dataset == "chalearn":
            # all utterances are in a single tsv file
            all_utts = pd.read_csv(current_file_location, sep='\t')

        return all_utts

    def read_in_google_transcribed_file(self):
        """
        read in google transcriptions
        """
        # same format for all google transcriptions
        if self.dataset == "mustard":
            transcribed = pd.read_csv(self.transcribed_file_path, names=["clip_id", "utterance", "confidence"], sep="\t")
            transcribed["clip_id"] = transcribed["clip_id"].str.replace(".wav", "")
        elif self.dataset == "meld":
            transcribed = pd.read_csv(self.transcribed_file_path, names=["DiaID_UttID", "Utterance", "confidence"], sep="\t")
            transcribed["DiaID_UttID"] = transcribed["DiaID_UttID"].str.replace("_2.wav", "")
        elif self.dataset == "chalearn":
            transcribed = pd.read_csv(self.transcribed_file_path, names=["file", "utterance", "confidence"], sep="\t")
            transcribed["file"] = transcribed["file"].str.replace(".wav", ".mp4")

        return transcribed

    def read_in_transcribed_file(self):
        """
        read in the transcriptions
        """
        # same format for all kaldi transcriptions
        if self.dataset == "mustard":
            transcribed = pd.read_csv(self.transcribed_file_path, names=["clip_id", "utterance"], sep="\t")
            # remove the file extension
            transcribed["clip_id"] = transcribed['clip_id'].str.replace(".wav", "")
        elif self.dataset == "meld":
            transcribed = pd.read_csv(self.transcribed_file_path, names=["DiaID_UttID", "Utterance"], sep="\t")
            # remove the file extension
            transcribed["DiaID_UttID"] = transcribed["DiaID_UttID"].str.replace(".wav", "")
        elif self.dataset == "chalearn":
            transcribed = pd.read_csv(self.transcribed_file_path, names=["file", "utterance"], sep="\t")
            # replace the file extension
            transcribed["file"] = transcribed["file"].str.replace(".wav", ".mp4")

        return transcribed

    def save_transcriptions(self, transcribed_data, current_files, save_name):
        """
        Save transcriptions alongside other info currently
        required for each dataset
        Saves to location self.location
        """
        sname = ""
        if save_name.endswith(".tsv"):
            sname = save_name
        elif save_name.endswith(".csv"):
            sname = save_name.split(".csv")[0]
        else:
            sname = f"{save_name}.tsv"

        # delete utterance from current_files
        current_files = current_files.loc[:, ~(current_files.columns.str.lower() == 'utterance')]

        # merge dfs on id
        if self.dataset == "meld":
            transcribed_data.rename(columns={'id': 'DiaID_UttID'}, inplace=True)
            transcribed_data = pd.merge(left=current_files, right=transcribed_data, how="left",
                                        left_on="DiaID_UttID", right_on="DiaID_UttID")
            # transcribed_data = transcribed_data.merge(current_files, on='DiaID_UttID')
        elif self.dataset == "mustard":
            transcribed_data.rename(columns={'id': 'clip_id'}, inplace=True)
            transcribed_data = pd.merge(left=current_files, right=transcribed_data, how="left",
                                        left_on="clip_id", right_on="clip_id")
            # transcribed_data = transcribed_data.merge(current_files, on='clip_id')
        elif self.dataset == "chalearn":
            transcribed_data.rename(columns={'id': 'file'}, inplace=True)
            transcribed_data = pd.merge(left=current_files, right=transcribed_data, how="left",
                                        left_on="file", right_on="file")
            # transcribed_data = transcribed_data.merge(current_files, on='file')

        transcribed_data.to_csv(f"{self.location}/{sname}", index=False,
                                sep="\t")


def combine_google_partial_transcripts(google_file, new_save_file):
    """
    Combine sequential partial transcripts produced by google
    Google splits anything large into multiple parts
    This just takes the first confidence value
        so data with multiple will not be correct
    todo: adjust confidence values
    """
    google_holder = {}
    print(f"preparing {google_file}")
    with open(google_file, 'r') as gfile:
        for line in gfile:
            line = line.strip().split("\t")
            if line[0] not in google_holder:
                google_holder[line[0]] = line[1:]
            else:
                google_holder[line[0]][0] += line[1]

    with open(new_save_file, 'w') as nfile:
        for k, v in google_holder.items():
            nfile.write(f"{k}\t{v[0]}\t{v[1]}\n")
    print(f"new file saved at: {new_save_file}")


if __name__ == "__main__":
    if sys.argv[1] == "mustard":
        # assumes that datasets are in the untracked 'data' directory
        mustard_location = "/Users/jculnan/datasets/multimodal_datasets/MUStARD"
        mustard_extensions = ""
        current_file_path = f"{mustard_location}/mustard_utts.tsv"
        transcribed_file = "mustard_16000_transcription.txt"

        mustard_organizer = OrganizeKaldiTranscriptions("MUStARD", mustard_location, mustard_extensions,
                                                        transcribed_file)

        # get current label file
        current_file = mustard_organizer.read_in_current_files(current_file_path)

        # transcribe data
        transcripts = mustard_organizer.read_in_transcribed_file()

        # save transcriptions
        mustard_organizer.save_transcriptions(transcripts, current_file, "mustard_kaldi.tsv")

    elif sys.argv[1] == "mustard-sphinx":
        # assumes that datasets are in the untracked 'data' directory
        mustard_location = "/Users/jculnan/datasets/multimodal_datasets/MUStARD"
        mustard_extensions = ""
        current_file_path = f"{mustard_location}/mustard_utts.tsv"
        transcribed_file = "mustard_sphinx.txt"

        mustard_organizer = OrganizeKaldiTranscriptions("MUStARD", mustard_location, mustard_extensions,
                                                        transcribed_file)

        # get current label file
        current_file = mustard_organizer.read_in_current_files(current_file_path)

        # transcribe data
        transcripts = mustard_organizer.read_in_transcribed_file()

        # save transcriptions
        mustard_organizer.save_transcriptions(transcripts, current_file, "mustard_sphinx.tsv")

    elif sys.argv[1] == "meld":
        # assumes that datasets are in the untracked 'data' directory
        meld_location = "/Users/jculnan/datasets/multimodal_datasets/MELD_formatted"

        meld_train_extensions = "train"
        meld_dev_extensions = "dev"
        meld_test_extensions = "test"

        transcribed_train = "meld_16000_train_transcription.txt"
        transcribed_dev = "meld_16000_dev_transcription.txt"
        transcribed_test = "meld_16000_test_transcription.txt"

        # transcribed_train = "meld_train_sphinx_16000.txt"
        # transcribed_dev = "meld_dev_sphinx_16000.txt"
        # transcribed_test = "meld_test_sphinx_16000.txt"

        current_train_path = f"{meld_location}/train/train_sent_emo.csv"
        current_dev_path = f"{meld_location}/dev/dev_sent_emo.csv"
        current_test_path = f"{meld_location}/test/test_sent_emo.csv"

        meld_train_organizer = OrganizeKaldiTranscriptions("MELD", meld_location, meld_train_extensions,
                                                             transcribed_train)
        meld_dev_organizer = OrganizeKaldiTranscriptions("MELD", meld_location, meld_dev_extensions,
                                                           transcribed_dev)
        meld_test_organizer = OrganizeKaldiTranscriptions("MELD", meld_location, meld_test_extensions,
                                                           transcribed_test)

        # get paths
        current_train_file = meld_train_organizer.read_in_current_files(current_train_path)
        current_dev_file = meld_dev_organizer.read_in_current_files(current_dev_path)
        current_test_file = meld_test_organizer.read_in_current_files(current_test_path)

        # transcribe data
        train_transcripts = meld_train_organizer.read_in_transcribed_file()
        dev_transcripts = meld_dev_organizer.read_in_transcribed_file()
        test_transcripts = meld_test_organizer.read_in_transcribed_file()

        # save transcriptions
        meld_train_organizer.save_transcriptions(train_transcripts, current_train_file, "train/meld_kaldi.tsv")
        meld_dev_organizer.save_transcriptions(dev_transcripts, current_dev_file, "dev/meld_kaldi.tsv")
        meld_test_organizer.save_transcriptions(test_transcripts, current_test_file, "test/meld_kaldi.tsv")

    elif sys.argv[1] == "chalearn":
        # assumes that datasets are in the untracked 'data' directory
        chalearn_location = "/Users/jculnan/datasets/multimodal_datasets/Chalearn"

        chalearn_train_extension = "train"
        chalearn_dev_extension = "val"
        chalearn_test_extension = "test"

        transcribed_train = "chalearn_16000_train_transcription.txt"
        transcribed_dev = "chalearn_16000_dev_transcription.txt"
        transcribed_test = "chalearn_16000_test_transcription.txt"

        current_train_path = f"{chalearn_location}/train/gold_and_utts.tsv"
        current_dev_path = f"{chalearn_location}/val/gold_and_utts.tsv"
        current_test_path = f"{chalearn_location}/test/gold_and_utts.tsv"

        chalearn_train_organizer = OrganizeKaldiTranscriptions("Chalearn", chalearn_location, chalearn_train_extension,
                                                               transcribed_train)
        current_train_file = chalearn_train_organizer.read_in_current_files(current_train_path)
        train_transcripts = chalearn_train_organizer.read_in_transcribed_file()

        # save transcriptions
        chalearn_train_organizer.save_transcriptions(train_transcripts, current_train_file, "train/chalearn_kaldi.tsv")

        chalearn_dev_organizer = OrganizeKaldiTranscriptions("Chalearn", chalearn_location, chalearn_dev_extension,
                                                             transcribed_dev)
        current_dev_file = chalearn_dev_organizer.read_in_current_files(current_dev_path)
        dev_transcripts = chalearn_dev_organizer.read_in_transcribed_file()

        # save transcriptions
        chalearn_dev_organizer.save_transcriptions(dev_transcripts, current_dev_file, "val/chalearn_kaldi.tsv")

        chalearn_test_organizer = OrganizeKaldiTranscriptions("Chalearn", chalearn_location, chalearn_test_extension,
                                                             transcribed_test)
        current_test_file = chalearn_test_organizer.read_in_current_files(current_test_path)
        test_transcripts = chalearn_test_organizer.read_in_transcribed_file()

        # save transcriptions
        chalearn_test_organizer.save_transcriptions(test_transcripts, current_test_file, "test/chalearn_kaldi.tsv")

    elif sys.argv[1] == "mustard-google":
        # assumes that datasets are in the untracked 'data' directory
        mustard_location = "/Users/jculnan/datasets/multimodal_datasets/MUStARD"
        mustard_extensions = ""
        current_file_path = f"{mustard_location}/mustard_utts.tsv"
        transcribed_file = "google_transcriptions_combined.txt"

        mustard_organizer = OrganizeKaldiTranscriptions("MUStARD", mustard_location, mustard_extensions,
                                                        transcribed_file)

        # get current label file
        current_file = mustard_organizer.read_in_current_files(current_file_path)

        # transcribe data
        transcripts = mustard_organizer.read_in_google_transcribed_file()

        # save transcriptions
        mustard_organizer.save_transcriptions(transcripts, current_file, "mustard_google.tsv")

    elif sys.argv[1] == "meld-google":
        # assumes that datasets are in the untracked 'data' directory
        meld_location = "/Users/jculnan/datasets/multimodal_datasets/MELD_formatted"

        meld_train_extensions = "train"
        meld_dev_extensions = "dev"
        meld_test_extensions = "test"

        transcribed_name = "google_transcriptions_combined.txt"

        current_train_path = f"{meld_location}/train/train_sent_emo.csv"
        current_dev_path = f"{meld_location}/dev/dev_sent_emo.csv"
        current_test_path = f"{meld_location}/test/test_sent_emo.csv"

        meld_train_organizer = OrganizeKaldiTranscriptions("MELD", meld_location, meld_train_extensions,
                                                             transcribed_name)
        meld_dev_organizer = OrganizeKaldiTranscriptions("MELD", meld_location, meld_dev_extensions,
                                                           transcribed_name)
        meld_test_organizer = OrganizeKaldiTranscriptions("MELD", meld_location, meld_test_extensions,
                                                           transcribed_name)

        # get paths
        current_train_file = meld_train_organizer.read_in_current_files(current_train_path)
        current_dev_file = meld_dev_organizer.read_in_current_files(current_dev_path)
        current_test_file = meld_test_organizer.read_in_current_files(current_test_path)

        # transcribe data
        train_transcripts = meld_train_organizer.read_in_google_transcribed_file()
        dev_transcripts = meld_dev_organizer.read_in_google_transcribed_file()
        test_transcripts = meld_test_organizer.read_in_google_transcribed_file()

        # save transcriptions
        meld_train_organizer.save_transcriptions(train_transcripts, current_train_file, "train/meld_google.tsv")
        meld_dev_organizer.save_transcriptions(dev_transcripts, current_dev_file, "dev/meld_google.tsv")
        meld_test_organizer.save_transcriptions(test_transcripts, current_test_file, "test/meld_google.tsv")

    elif sys.argv[1] == "chalearn-google":
        # assumes that datasets are in the untracked 'data' directory
        chalearn_location = "/Users/jculnan/datasets/multimodal_datasets/Chalearn"

        chalearn_train_extension = "train"
        chalearn_dev_extension = "val"
        chalearn_test_extension = "test"

        transcribed_name = "google_transcriptions_combined.txt"

        current_train_path = f"{chalearn_location}/train/gold_and_utts.tsv"
        current_dev_path = f"{chalearn_location}/val/gold_and_utts.tsv"
        current_test_path = f"{chalearn_location}/test/gold_and_utts.tsv"

        chalearn_train_organizer = OrganizeKaldiTranscriptions("Chalearn", chalearn_location, chalearn_train_extension,
                                                               transcribed_name)
        current_train_file = chalearn_train_organizer.read_in_current_files(current_train_path)
        train_transcripts = chalearn_train_organizer.read_in_google_transcribed_file()

        # save transcriptions
        chalearn_train_organizer.save_transcriptions(train_transcripts, current_train_file, "train/chalearn_google.tsv")

        chalearn_dev_organizer = OrganizeKaldiTranscriptions("Chalearn", chalearn_location, chalearn_dev_extension,
                                                             transcribed_name)
        current_dev_file = chalearn_dev_organizer.read_in_current_files(current_dev_path)
        dev_transcripts = chalearn_dev_organizer.read_in_google_transcribed_file()

        # save transcriptions
        chalearn_dev_organizer.save_transcriptions(dev_transcripts, current_dev_file, "val/chalearn_google.tsv")

        chalearn_test_organizer = OrganizeKaldiTranscriptions("Chalearn", chalearn_location, chalearn_test_extension,
                                                             transcribed_name)
        current_test_file = chalearn_test_organizer.read_in_current_files(current_test_path)
        test_transcripts = chalearn_test_organizer.read_in_google_transcribed_file()

        # save transcriptions
        chalearn_test_organizer.save_transcriptions(test_transcripts, current_test_file, "test/chalearn_google.tsv")

    elif sys.argv[1] == "combine_google":
        # mustard
        # assumes that datasets are in the untracked 'data' directory
        mustard_location = "/Users/jculnan/datasets/multimodal_datasets/MUStARD"
        mustard_extensions = ""
        current_file_path = f"{mustard_location}/mustard_utts.tsv"
        transcribed_file = "google_transcriptions.txt"

        combine_google_partial_transcripts(f"{mustard_location}/{transcribed_file}",
                                           f"{mustard_location}/google_transcriptions_combined.txt")

        # meld
        meld_location = "/Users/jculnan/datasets/multimodal_datasets/MELD_formatted"

        meld_train_extensions = "train"
        meld_dev_extensions = "dev"
        meld_test_extensions = "test"

        transcribed_name = "google_transcriptions.txt"

        combine_google_partial_transcripts(f"{meld_location}/{meld_train_extensions}/{transcribed_name}",
                                           f"{meld_location}/{meld_train_extensions}/google_transcriptions_combined.txt")

        combine_google_partial_transcripts(f"{meld_location}/{meld_dev_extensions}/{transcribed_name}",
                                           f"{meld_location}/{meld_dev_extensions}/google_transcriptions_combined.txt")

        combine_google_partial_transcripts(f"{meld_location}/{meld_test_extensions}/{transcribed_name}",
                                           f"{meld_location}/{meld_test_extensions}/google_transcriptions_combined.txt")


        # chalearn
        # assumes that datasets are in the untracked 'data' directory
        chalearn_location = "/Users/jculnan/datasets/multimodal_datasets/Chalearn"

        chalearn_train_extension = "train"
        chalearn_dev_extension = "val"
        chalearn_test_extension = "test"

        transcribed_name = "google_transcriptions.txt"

        combine_google_partial_transcripts(f"{chalearn_location}/{chalearn_train_extension}/{transcribed_name}",
                                           f"{chalearn_location}/{chalearn_train_extension}/google_transcriptions_combined.txt")

        combine_google_partial_transcripts(f"{chalearn_location}/{chalearn_dev_extension}/{transcribed_name}",
                                           f"{chalearn_location}/{chalearn_dev_extension}/google_transcriptions_combined.txt")

        combine_google_partial_transcripts(f"{chalearn_location}/{chalearn_test_extension}/{transcribed_name}",
                                           f"{chalearn_location}/{chalearn_test_extension}/google_transcriptions_combined.txt")