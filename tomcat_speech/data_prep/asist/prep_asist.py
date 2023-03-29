
import torch
from sklearn.utils import compute_class_weight

import pickle
import os

from tomcat_speech.data_prep.utils.data_prep_helpers import (
    make_glove_dict,
    Glove,
)
from tomcat_speech.data_prep.prep_data import SelfSplitPrep


def prep_asist_data(
    data_path="../../asist_data2/overall_sent-emo.csv",
    feature_set="IS13",
    transcription_type="gold",
    embedding_type="distilbert",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    num_train_ex=None,
    include_spectrograms=False,
):
    # load glove
    if embedding_type.lower() == "glove":
        glove_dict = make_glove_dict(glove_filepath)
        glove = Glove(glove_dict)
    else:
        glove = None

    # holder for name of file containing utterance info
    utts_name = "overall_sent-emo.csv"

    # create instance of StandardPrep class
    asist_prep = SelfSplitPrep(
        data_type="asist",
        data_path=data_path,
        feature_set=feature_set,
        utterance_fname=utts_name,
        glove=glove,
        use_cols=features_to_use,
        as_dict=as_dict,
        avg_acoustic_data=avg_acoustic_data,
        custom_feats_file=custom_feats_file,
        bert_type=embedding_type,
        include_spectrograms=include_spectrograms
    )

    # get train, dev, test data
    train_data, dev_data, test_data = asist_prep.get_data_folds()

    # get train ys
    train_ys = [item['ys'] for item in train_data]

    # get updated class weights using train ys
    class_weights = get_updated_class_weights_multicat(train_ys)

    return train_data, dev_data, test_data, class_weights


def get_updated_class_weights_multicat(train_ys):
    """
    Get updated class weights
    Because DataPrep assumes you only enter train set
    :return:
    """
    all_task_weights = []
    # get class weights for each task
    for i in range(len(train_ys[0])):
        labels = [int(y[i]) for y in train_ys]
        classes = sorted(list(set(labels)))
        weights = compute_class_weight("balanced", classes=classes, y=labels)
        weights = torch.tensor(weights, dtype=torch.float)
        all_task_weights.append(weights)

    return all_task_weights


def save_asist_data(
    dataset,
    save_path,
    data_path,
    feature_set,
    glove_path,
    emb_type,
    feats_to_use=None,
    data_as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    include_spectrograms=False,
    num_partitions=1
):
    """
    Save partitioned data in pickled format
    :param dataset: the string name of dataset to use
    :param save_path: path where you want to save pickled data
    :param data_path: path to the data
    :param feature_set: IS09-13
    :param transcription_type: Gold, Google, Kaldi, Sphinx
    :param glove_path: path to glove file
    :param emb_type: whether to use glove or distilbert
    :param feats_to_use: list of features, if needed
    :param pred_type: type of predictions, for mosi and firstimpr
    :param zip: whether to save as a bz2 compressed file
    :param data_as_dict: whether each datapoint saves as a dict
    :return:
    """
    dataset = dataset.lower()

    # make sure the full save path exists; if not, create it
    os.system(f'if [ ! -d "{save_path}" ]; then mkdir -p {save_path}; fi')

    train_ds, dev_ds, test_ds, clss_weights = prep_asist_data(
        data_path,
        feature_set,
        emb_type,
        glove_path,
        feats_to_use,
        data_as_dict,
        avg_acoustic_data,
        custom_feats_file,
        include_spectrograms
    )

    # use custom feats set instead of ISXX in save name
    #   if custom feats are used
    if custom_feats_file is not None:
        feature_set = custom_feats_file.split(".")[0]

    if data_as_dict:
        dtype = "dict"
    else:
        dtype = "list"

    if include_spectrograms:
        train_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_spec_train"
        dev_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_spec_dev"
        test_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_spec_test"
        wts_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_spec_clsswts"
    else:
        train_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_train"
        dev_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_dev"
        test_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_test"
        wts_save_name = f"{save_path}/{dataset}_{feature_set}_{emb_type}_{dtype}_clsswts"

    if num_partitions == 3:
        pickle.dump(train_ds,open(f"{train_save_name}.pickle", "wb"))
        pickle.dump(dev_ds, open(f"{dev_save_name}.pickle", "wb"))
        pickle.dump(test_ds, open(f"{test_save_name}.pickle", "wb"))
        pickle.dump(clss_weights, open(f"{wts_save_name}.pickle", "wb"))
    elif num_partitions == 1:
        data = train_ds + dev_ds + test_ds
        pickle.dump(data, open(f"{test_save_name}.pickle", 'wb'))


if __name__ == "__main__":
    save_path = "../../dataset/pickled_data"
    # data_path = "../../PROJECTS/ToMCAT/Evaluating_modelpredictions/data_from_speechAnalyzer/used_for_evaluating_model_results"
    data_path = "../../study3_data_to_annotate"
    emb_type = "glove"
    glove_path = "../../datasets/glove/glove.subset.300d.txt"
    dict_data = True
    avg_feats = True
    with_spec = False
    partitions = 1

    save_asist_data("asist",
                    save_path=save_path,
                    data_path=data_path,
                    feature_set="IS13",
                    glove_path=glove_path,
                    emb_type=emb_type,
                    data_as_dict=dict_data,
                    avg_acoustic_data=avg_feats,
                    include_spectrograms=with_spec,
                    num_partitions=partitions)
