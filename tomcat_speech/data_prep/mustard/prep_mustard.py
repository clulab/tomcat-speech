from tomcat_speech.data_prep.prep_data import SelfSplitPrep, get_updated_class_weights
from tomcat_speech.data_prep.utils.data_prep_helpers import (
    Glove,
    make_glove_dict,
)


def prep_mustard_data(
    data_path="../../datasets/multimodal_datasets/mustard",
    feature_set="IS13",
    transcription_type="gold",
    embedding_type="distilbert",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    include_spectrograms=False,
):
    """
    Prepare pickle files for MELD data
    :param data_path: the string path to directory containing the dataset
    :param feature_set: the acoustic feature set to use (generally IS13)
        could be IS09, IS10, IS11, IS12, IS13
    :param transcription_type: string name of transcription type 'gold'
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
    if transcription_type.lower() == "gold":
        utts_name = "mustard_utts.tsv"
    else:
        utts_name = f"mustard_{transcription_type.lower()}.tsv"

    # create instance of StandardPrep class
    mustard_prep = SelfSplitPrep(
        data_type="mustard",
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
    train_data, dev_data, test_data = mustard_prep.get_data_folds()

    # get train ys
    if as_dict:
        train_ys = [int(item["ys"][0]) for item in train_data]
    else:
        train_ys = [int(item[4]) for item in train_data]

    # get updated class weights using train ys
    class_weights = get_updated_class_weights(train_ys)

    return train_data, dev_data, test_data, class_weights
