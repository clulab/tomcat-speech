# prepare your data using the multimodal_data_preprocessing repo
# this assumes that the repo multimodal_data_preprocessing is in the same parent directory as mmml
# to change the directory, add the path to multimodal_data_preprocessing to your paths

import argparse
import sys
import os

parser = argparse.ArgumentParser(description="Select the datasets to preprocess and their manner of preprocessing")

parser.add_argument("--datasets", help="the datasets to preprocess",
                    default=["meld", "firstimpr", "ravdess",
                             "cdc", "mustard", "mosi"],
                    nargs="+")

parser.add_argument("--embedding_type", help="the embedding type (GloVe or DistilBERT)",
                    default="distilbert")

parser.add_argument("--save_location", help="the directory where pickled data should be saved",
                    default="data")

parser.add_argument("--load_location", help="the location of the raw datasets",
                    default="../multimodal_data_preprocessing/data")

parser.add_argument("--glove_location", help="the location of a glove file, if using",
                    nargs="?")

parser.add_argument("--feature_set", help="the acoustic feature set to select in openSMILE (IS09...IS13)",
                    default="IS13")

args = parser.parse_args()

if __name__ == "__main__":
    emb_type = args.embedding_type
    save_path = args.save_location
    load_path = args.load_location
    if args.glove_location:
        glove_path = args.glove_location
    else:
        print("no GloVe selected")
        glove_path = ""
    f_set = args.feature_set
    all_datasets = args.datasets

    # add multimodal data preprocessing code to path
    current = os.path.abspath(os.getcwd())
    parent = os.path.dirname(current)

    sys.path.append(f"{parent}/multimodal_data_preprocessing")

    print(sys.path)

    # todo: make sure this works after sys path append
    from preprocessing_scripts import save_partitioned_data

    for dataset in all_datasets:
        if dataset.lower() == "meld":
            dset_path = f"{load_path}/MELD_formatted"
        elif dataset.lower() == "mustard":
            dset_path = f"{load_path}/MUStARD"
        elif dataset.lower() == "cdc":
            dset_path = f"{load_path}/CDC"
        elif dataset.lower() == "mosi" or dataset.lower == "cmu_mosi":
            dset_path = f"{load_path}/CMU_MOSI"
        elif dataset.lower() == "firstimpr":
            dset_path = f"{load_path}/Chalearn"
        elif dataset.lower() == "ravdess":
            dset_path = f"{load_path}/RAVDESS"
        else:
            exit(f"Dataset {dataset} not recognized")

        save_partitioned_data.save_partitioned_data(dataset=dataset,
                                                    save_path=save_path,
                                                    data_path=dset_path,
                                                    feature_set=f_set,
                                                    transcription_type="gold",
                                                    glove_path=glove_path,
                                                    emb_type=emb_type,
                                                    data_as_dict=True)
