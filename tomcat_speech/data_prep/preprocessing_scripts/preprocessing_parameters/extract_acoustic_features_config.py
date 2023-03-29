# config file for extract_acoustic_features.py
# use this file to set paths relevant for extracting acoustic features
# for the datasets of interest
# you may extract features as openSMILE acoustic features
# or as wav2vec2 spectrogram features

# set the datasets of interest as a list of strings
# 'asist', 'cdc', 'mosi', 'firstimpr', 'meld', 'mustard', 'ravdess'
datasets = ["asist"]

# set the path to the datasets
# this assumes that all datasets are in the same base directory
path_to_datasets = "../datasets/multimodal_datasets"

# set the type of features to extract
# can be IS09-IS13 and/or spectrogram features
features_of_interest = ["IS13"]