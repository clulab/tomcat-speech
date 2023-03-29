
from tomcat_speech.data_prep.utils.data_saving_and_loading_helpers import prep_data

import pickle
import bz2
import os


def save_data_components(
    dataset,
    save_path,
    data_path,
    feature_set,
    transcription_type,
    glove_path,
    emb_type,
    feats_to_use=None,
    pred_type=None,
    use_zip=False,
    custom_feats_file=None,
    selected_ids=None,
):
    """
    Save partitioned data in pickled format
    :param dataset: the string name of dataset to use
        'asist', 'cdc', 'mosi', 'firstimpr', 'meld', 'mustard', 'ravdess'
    :param save_path: string path where you want to save pickled data
    :param data_path: string path to the data
    :param feature_set: acoustic feature set to use; usually 'IS13'
        'IS09', 'IS10', 'IS11', 'IS12', 'IS13'
    :param transcription_type: Generally 'gold' unless testing new ASR data
        'gold', 'google', 'kaldi', 'sphinx' for 2021 paper
    :param glove_path: string path to glove file
    :param emb_type: embedding type to use
        'glove', 'distilbert', 'bert', 'roberta'
    :param feats_to_use: None or list of acoustic features to use from openSMILE extraction
        None uses the whole set of extracted acoustic features
    :param pred_type: type of predictions, for mosi and firstimpr
        mosi: 'regression', 'classification' (7-class), 'ternary' (3-class classification)
        firstimpr: 'max_class' (dominant trait), 'binary' (high-low per trait),
        'ternary' (high-med-low per trait)
    :param use_zip: whether to save as a bz2 compressed file
    :param custom_feats_file: the string path to a file containing custom acoustic features
        usually NOT used unless you are testing out a new set of acoustic features
        which you have extracted on your own
    :param selected_ids: None if generating data randomly or a list of 3 lists
        One with message IDs for train partition, one for dev, one for test
        This is needed only for datasets that are not pre-partitioned if you
        wish to ensure that you have a specific split of the data
    """
    dataset = dataset.lower()

    # make sure the full save path exists; if not, create it
    os.system(f'if [ ! -d "{save_path}" ]; then mkdir -p {save_path}; fi')

    train_ds, dev_ds, test_ds, clss_weights = prep_data(
        dataset,
        data_path,
        feature_set,
        transcription_type,
        glove_path,
        emb_type,
        feats_to_use,
        pred_type,
        True,  # data must be dict to use this saving format
        False,  # acoustic data must not be averaged
        custom_feats_file,
        selected_ids=selected_ids,
        include_spectrograms=False
    )
    # save class weights
    if use_zip:
        pickle.dump(clss_weights, bz2.BZ2File(f"{save_path}/{dataset}_clsswts.bz2", "wb"))
    else:
        pickle.dump(clss_weights, open(f"{save_path}/{dataset}_clsswts.pickle", "wb"))

    all_data = [('train', train_ds),
                ('dev', dev_ds),
                ('test', test_ds)]

    for partition_tuple in all_data:
        # get name of partition
        partition_name = partition_tuple[0]
        partition = partition_tuple[1]
        # get spec data + audio_ids
        spec_data = get_specific_fields(partition, "spec")
        spec_save_name = f"{save_path}/{dataset}_spec_{partition_name}"

        # get acoustic data + audio_ids
        acoustic_data = get_specific_fields(partition, "acoustic")
        # use custom feats set instead of ISXX in save name
        #   if custom feats are used
        if custom_feats_file is not None:
            feature_set = custom_feats_file.split(".")[0]
        acoustic_save_name = f"{save_path}/{dataset}_{feature_set}_{partition_name}"

        # get utt data + audio_ids
        utt_data = get_specific_fields(partition, "utt")
        utt_save_name = f"{save_path}/{dataset}_{emb_type}_{partition_name}"

        # get ys data + audio_ids
        ys_data = get_specific_fields(partition, "ys")
        ys_save_name = f"{save_path}/{dataset}_ys_{partition_name}"

        # save
        if use_zip:
            pickle.dump(spec_data, bz2.BZ2File(f"{spec_save_name}.bz2", "wb"))
            pickle.dump(acoustic_data, bz2.BZ2File(f"{acoustic_save_name}.bz2", "wb"))
            pickle.dump(utt_data, bz2.BZ2File(f"{utt_save_name}.bz2", "wb"))
            pickle.dump(ys_data, bz2.BZ2File(f"{ys_save_name}.bz2", "wb"))
        else:
            pickle.dump(spec_data,open(f"{spec_save_name}.pickle", "wb"))
            pickle.dump(acoustic_data, open(f"{acoustic_save_name}.pickle", "wb"))
            pickle.dump(utt_data, open(f"{utt_save_name}.pickle", "wb"))
            pickle.dump(ys_data, open(f"{ys_save_name}.pickle", "wb"))


def get_specific_fields(data, field_type, fields=None):
    """
    Partition the data based on a set of keys
    :param data: The dataset
    :param field_type: 'spec', 'acoustic', 'utt', 'ys', or 'other'
    :param fields: if specific fields are given, use this instead of
        field type to get portions of data
    :return: The subset of data with these fields
    """
    sub_data = []
    if fields is not None:
        for item in data:
            sub_data.append({key: value for key, value in item.items() if key in fields})
    else:
        if field_type.lower() == "spec":
            keys = ["x_spec", "spec_length", "audio_id"]
        elif field_type.lower() == "acoustic":
            keys = ["x_acoustic", "acoustic_length", "audio_id"]
        elif field_type.lower() == "utt":
            keys = ["x_utt", "utt_length", "audio_id"]
        elif field_type.lower() == "ys":
            keys = ["ys", "audio_id"]
        else:
            exit("Field type not listed, and no specific fields given")

        for item in data:
            sub_data.append({key: value for key, value in item.items() if key in keys})

    return sub_data


if __name__ == "__main__":
    # read in the config file
    # to change what datasets you preprocess and how you do it
    # make alterations to save_modalities_separately_config
    # and not this file
    from tomcat_speech.data_prep.preprocessing_scripts.preprocessing_parameters import preprocess_datasets_config as config

    for dataset in config.datasets:
        if dataset == "mosi":
            save_data_components(
                dataset,
                config.save_path,
                config.mosi_path,
                config.feature_set,
                config.transcription_type,
                config.glove_path,
                pred_type=config.mosi_pred_type,
                emb_type=config.emb_type,
                custom_feats_file=config.custom_feats_file["mosi"] if "mosi" in config.custom_feats_file.keys() else None,
            )
        elif dataset == "firstimpr":
            save_data_components(
                dataset,
                config.save_path,
                config.firstimpr_path,
                config.feature_set,
                config.transcription_type,
                config.glove_path,
                pred_type=config.firstimpr_pred_type,
                emb_type=config.emb_type,
                custom_feats_file=config.custom_feats_file["firstimpr"] if "firstimpr" in config.custom_feats_file.keys() else None,
            )
        elif dataset == "cdc":
            save_data_components(
                dataset,
                config.save_path,
                config.cdc_path,
                config.feature_set,
                config.transcription_type,
                config.glove_path,
                emb_type=config.emb_type,
                custom_feats_file=config.custom_feats_file["cdc"] if "cdc" in config.custom_feats_file.keys() else None,
            )
        elif dataset == "meld":
            save_data_components(
                dataset,
                config.save_path,
                config.meld_path,
                config.feature_set,
                config.transcription_type,
                config.glove_path,
                emb_type=config.emb_type,
                custom_feats_file=config.custom_feats_file["meld"] if "meld" in config.custom_feats_file.keys() else None,
            )
        elif dataset == "mustard":
            save_data_components(
                dataset,
                config.save_path,
                config.mustard_path,
                config.feature_set,
                config.transcription_type,
                config.glove_path,
                emb_type=config.emb_type,
                custom_feats_file=config.custom_feats_file["mustard"] if "mustard" in config.custom_feats_file.keys() else None,
            )
        elif dataset == "ravdess":
            save_data_components(
                dataset,
                config.save_path,
                config.ravdess_path,
                config.feature_set,
                config.transcription_type,
                config.glove_path,
                emb_type=config.emb_type,
                custom_feats_file=config.custom_feats_file["ravdess"] if "ravdess" in config.custom_feats_file.keys() else None,
                selected_ids=config.selected_ids_paths["ravdess"] if "ravdess" in config.selected_ids_paths.keys() else None,
            )
        elif dataset == "asist":
            save_data_components(
                dataset,
                config.save_path,
                config.asist_path,
                config.feature_set,
                config.transcription_type,
                config.glove_path,
                emb_type=config.emb_type,
                custom_feats_file=config.custom_feats_file["asist"] if "asist" in config.custom_feats_file.keys() else None,
            )
