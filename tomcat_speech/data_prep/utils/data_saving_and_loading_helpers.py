from tomcat_speech.data_prep.asist.prep_asist import prep_asist_data
from tomcat_speech.data_prep.cdc.prep_cdc import *
from tomcat_speech.data_prep.cmu_mosi.prep_mosi import *
from tomcat_speech.data_prep.firstimpr.prep_firstimpr import *
from tomcat_speech.data_prep.meld.prep_meld import *
from tomcat_speech.data_prep.mustard.prep_mustard import *
from tomcat_speech.data_prep.ravdess.prep_ravdess import *

from torch.utils.data import Dataset


def prep_data(
    dataset,
    data_path,
    feature_set,
    transcription_type,
    glove_path,
    emb_type,
    feats_to_use,
    pred_type=None,
    data_as_dict=False,
    avg_acoustic_data=False,
    custom_feats_file=None,
    selected_ids=None,
    include_spectrograms=False,
):
    """
    Prepare data for a given dataset
    :param dataset: the string name of dataset to use
        'asist', 'cdc', 'mosi', 'firstimpr', 'meld', 'mustard', 'ravdess'
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
    :param data_as_dict: whether saved data points will be dicts (or lists)
    :param avg_acoustic_data: whether to average acoustic features
    :param custom_feats_file: the string path to a file containing custom acoustic features
        usually NOT used unless you are testing out a new set of acoustic features
        which you have extracted on your own
    :param selected_ids: None if generating data randomly or a list of 3 lists
        One with message IDs for train partition, one for dev, one for test
        This is needed only for datasets that are not pre-partitioned if you
        wish to ensure that you have a specific split of the data
    :param include_spectrograms: whether to include spectrograms
    :return: train data, dev data, test data, and class weights
    """
    dataset = dataset.lower()

    print("-------------------------------------------")
    print(f"Starting dataset prep for {dataset}")
    print("-------------------------------------------")

    if dataset == "cdc":
        train, dev, test, weights = prep_cdc_data(
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            include_spectrograms=include_spectrograms,
        )
    elif dataset == "mosi" or dataset == "cmu_mosi" or dataset == "cmu-mosi":
        train, dev, test, weights = prep_mosi_data(
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            pred_type,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            include_spectrograms=include_spectrograms,
        )
    elif dataset == "firstimpr" or dataset == "chalearn":
        train, dev, test, weights = prep_firstimpr_data(
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            pred_type,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            include_spectrograms=include_spectrograms,
        )
    elif dataset == "meld":
        train, dev, test, weights = prep_meld_data(
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            include_spectrograms=include_spectrograms,
        )
    elif dataset == "mustard":
        train, dev, test, weights = prep_mustard_data(
            data_path,
            feature_set,
            transcription_type,
            emb_type,
            glove_path,
            feats_to_use,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            include_spectrograms=include_spectrograms,
        )
    elif dataset == "ravdess":
        train, dev, test, weights = prep_ravdess_data(
            data_path,
            feature_set,
            emb_type,
            glove_path,
            feats_to_use,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            selected_ids=selected_ids,
            include_spectrograms=include_spectrograms,
        )
    elif dataset == "asist":
        train, dev, test, weights = prep_asist_data(
            data_path,
            feature_set,
            emb_type,
            glove_path,
            feats_to_use,
            as_dict=data_as_dict,
            avg_acoustic_data=avg_acoustic_data,
            custom_feats_file=custom_feats_file,
            include_spectrograms=include_spectrograms,
        )

    return train, dev, test, weights


class DatumListDataset(Dataset):
    """
    A dataset to hold a list of datums
    """

    def __init__(self, data_list, data_type="meld_emotion", class_weights=None):
        self.data_list = data_list
        self.data_type = data_type
        # todo: add task number

        self.class_weights = class_weights

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        """
        item (int) : the index to a data point
        """
        return self.data_list[item]

    def targets(self):
        if (
            self.data_type == "meld_emotion"
            or self.data_type == "mustard"
            or self.data_type == "ravdess_emotion"
        ):
            for datum in self.data_list:
                yield datum[4]
        elif (
            self.data_type == "meld_sentiment" or self.data_type == "ravdess_intensity"
        ):
            for datum in self.data_list:
                yield datum[5]
