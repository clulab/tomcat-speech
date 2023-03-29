from tomcat_speech.data_prep.prep_data import *
from tomcat_speech.data_prep.utils.data_prep_helpers import Glove, make_glove_dict, get_data_samples


def prep_firstimpr_data(
    data_path="../../datasets/multimodal_datasets/FirstImpr",
    feature_set="IS13",
    transcription_type="gold",
    embedding_type="distilbert",
    glove_filepath="../asist-speech/data/glove.short.300d.punct.txt",
    features_to_use=None,
    pred_type="max_class",
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
    if transcription_type.lower() == "gold":
        utts_name = "gold_and_utts.tsv"
    else:
        utts_name = f"chalearn_{transcription_type.lower()}.tsv"

    # create instance of StandardPrep class
    firstimpr_prep = StandardPrep(
        data_type="firstimpr",
        data_path=data_path,
        feature_set=feature_set,
        utterance_fname=utts_name,
        glove=glove,
        transcription_type=transcription_type,
        use_cols=features_to_use,
        avg_acoustic_data=avg_acoustic_data,
        custom_feats_file=custom_feats_file,
        bert_type=embedding_type,
        include_spectrograms=include_spectrograms
    )

    # add the prediction type, since first impressions can have several
    firstimpr_prep.train_prep.add_pred_type(pred_type)
    firstimpr_prep.dev_prep.add_pred_type(pred_type)
    firstimpr_prep.test_prep.add_pred_type(pred_type)

    # get train, dev, test data
    print("Now preparing training data")
    train_data = firstimpr_prep.train_prep.combine_xs_and_ys(as_dict=as_dict)
    # get class weights
    class_weights = firstimpr_prep.train_prep.class_weights

    del firstimpr_prep.train_prep
    print("Now preparing development data")
    dev_data = firstimpr_prep.dev_prep.combine_xs_and_ys(as_dict=as_dict)

    del firstimpr_prep.dev_prep
    print("Now preparing test data")
    test_data = firstimpr_prep.test_prep.combine_xs_and_ys(as_dict=as_dict)
    del firstimpr_prep.test_prep

    if num_train_ex:
        train_data = get_data_samples(train_data, num_train_ex)

    return train_data, dev_data, test_data, class_weights


def convert_ys(ys, conversion="high-low", mean_y=None, one_third=None, two_thirds=None):
    """
    Convert a set of ys into binary high-low labels
    or ternary high-medium-low labels
    Uses the mean for binary and one-third and two-third
        markers for ternary labels
    """
    new_ys = []
    if conversion == "high-low" or conversion == "binary":
        for item in ys:
            if mean_y:
                if item >= mean_y:
                    new_ys.append(1)
                else:
                    new_ys.append(0)
            else:
                if item >= 0.5:
                    new_ys.append(1)
                else:
                    new_ys.append(0)
    elif conversion == "high-med-low" or conversion == "ternary":
        if one_third and two_thirds:
            for item in ys:
                if item >= two_thirds:
                    new_ys.append(2)
                elif one_third <= item < two_thirds:
                    new_ys.append(1)
                else:
                    new_ys.append(0)
        else:
            for item in ys:
                if item >= 0.67:
                    new_ys.append(2)
                elif 0.34 <= item < 0.67:
                    new_ys.append(1)
                else:
                    new_ys.append(0)
    return new_ys
