# use this script to extract acoustic features from your datasets
# acoustic features extracted are saved in new subdirectories
# within the original dataset directories
# these features are then read in when preprocessing the data
# this script assumes that all the datasets you will use
#   exist in the same base directory
# to select which datasets to extract features for, and which
# features to extract, alter extract_acoustic_features_config.py
# you should not need to alter this file directly

from tomcat_speech.data_prep.utils.audio_extraction import run_feature_extraction


def extract_acoustic_features(datasets_list, base_path, feature_set):
    """
    Extracts acoustic features using openSMILE
    :param datasets_list: A list of datasets used
    :param base_path: The string name of the path to the directory
        that contains all datasets
    :param feature_set: The set of features to extract
    """
    for dataset in datasets_list:
        dataset = dataset.lower()
        if dataset == "cdc":
            cdc_path = f"{base_path}/Columbia_deception_corpus"
            run_feature_extraction(
                f"{cdc_path}/wav", feature_set, f"{cdc_path}/{feature_set}"
            )
        elif dataset == "firstimpr":
            firstimpr_path = f"{base_path}/FirstImpr"
            run_feature_extraction(
                f"{firstimpr_path}/train/wav",
                feature_set,
                f"{firstimpr_path}/train/{feature_set}",
            )
            run_feature_extraction(
                f"{firstimpr_path}/val/wav",
                feature_set,
                f"{firstimpr_path}/val/{feature_set}",
            )
            run_feature_extraction(
                f"{firstimpr_path}/test/wav",
                feature_set,
                f"{firstimpr_path}/test/{feature_set}",
            )
        elif dataset == "meld":
            meld_path = f"{base_path}/MELD_formatted"
            meld_train_path = f"{meld_path}/train/train_audio_mono"
            meld_dev_path = f"{meld_path}/dev/dev_audio_mono"
            meld_test_path = f"{meld_path}/test/test_audio_mono"
            run_feature_extraction(
                meld_train_path, feature_set, f"{meld_path}/train/{feature_set}"
            )
            run_feature_extraction(
                meld_dev_path, feature_set, f"{meld_path}/dev/{feature_set}"
            )
            run_feature_extraction(
                meld_test_path, feature_set, f"{meld_path}/test/{feature_set}"
            )
        elif dataset == "mosi" or dataset == "cmu-mosi":
            mosi_path = f"{base_path}/CMU_MOSI"
            run_feature_extraction(
                f"{mosi_path}/Audio/WAV_16000/Segmented",
                feature_set,
                f"{mosi_path}/{feature_set}",
            )
        elif dataset == "ravdess":
            ravdess_path = f"{base_path}/RAVDESS_Speech"
            run_feature_extraction(
                f"{ravdess_path}/All_actors",
                feature_set,
                f"{ravdess_path}/{feature_set}",
            )
        elif dataset == "asist":
            asist_path = base_path
            run_feature_extraction(
                f"{asist_path}/split", feature_set, f"{asist_path}/{feature_set}"
            )


if __name__ == "__main__":
    from tomcat_speech.data_prep.preprocessing_scripts.preprocessing_parameters import (
        extract_acoustic_features_config as config,
    )

    for feature in config.features_of_interest:
        extract_acoustic_features(config.datasets, config.path_to_datasets, feature)
