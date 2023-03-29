import torch

from sklearn.utils import compute_class_weight

from tomcat_speech.data_prep.utils.data_prep_helpers import (
    make_glove_dict,
    Glove,
)
from tomcat_speech.data_prep.prep_data import SelfSplitPrep


def prep_asist_data(
    data_path="../../asist_data2/overall_sent-emo.csv",
    feature_set="IS13",
    embedding_type="distilbert",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    as_dict=True,
    avg_acoustic_data=False,
    custom_feats_file=None,
    include_spectrograms=False,
):
    """
    Prepare pickle files for ASIST data
    :param data_path: the string path to file containing messages and gold labels
    :param feature_set: the acoustic feature set to use (generally IS13)
    :param embedding_type: 'bert', 'roberta', 'distilbert', 'glove'
    :param glove_filepath: the string path to a txt file containing GloVe
    :param features_to_use: None or a list of specific acoustic features
        if a list, these features are pulled from the opensmile output by name
    :param as_dict: whether to save data as list of dicts (vs lists)
        Our models expect dict data as of 2023.03.28
    :param avg_acoustic_data: whether to average over acoustic features
    :param custom_feats_file: None or the string name of a file containing
        pre-generated custom acoustic features
    :param include_spectrograms: whether to use spectrograms as part of the dataset
    """
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
        include_spectrograms=include_spectrograms,
    )

    # get train, dev, test data
    train_data, dev_data, test_data = asist_prep.get_data_folds()

    # get train ys
    train_ys = [item["ys"] for item in train_data]

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
