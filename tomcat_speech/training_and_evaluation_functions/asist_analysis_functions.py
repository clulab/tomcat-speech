# this file was used with study 3 online analysis
# it is not needed for offline research or work with MultiCAT
import random
import json
import re
import numpy as np
import torch

from tomcat_speech.data_prep.asist.asist_dataset_creation import AsistDataset
from tomcat_speech.training_and_evaluation_functions.train_and_test_without_gold_labels import (
    multitask_predict_without_gold_labels,
)
from tomcat_speech.models.multimodal_models import MultitaskModel
from tomcat_speech.data_prep.utils.data_prep_helpers import DatumListDataset

import pandas as pd


def read_in_aligned_json(input_aligned_json):
    """
    Read in a json file containing utterances with word-aligned data
    Each utterance is a separate json file
    Each utterance is separated by a single blank line
    """
    # holder for all json objects
    list_of_json = []

    # for file in files (in case more than one)
    for json_file in input_aligned_json:
        # open the file with multiple jsons
        with open(json_file, "r") as j_file:
            # get the data from all jsons in this file
            all_jsons = decode_stacked(j_file)

            list_of_json.extend(all_jsons)

    return list_of_json


# make new version with Glove Obj


def predict_with_model(list_of_json_objs, trained_model, glove, device, params):
    # set random seed
    seed = params.model_params.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # decide if you want to use avgd feats
    avgd_acoustic = params.model_params.avgd_acoustic or params.model_params.add_avging

    # get acoustic dict and utts dict
    acoustic_dict = get_data_from_json(list_of_json_objs, avgd_acoustic)
    print("Acoustic dict created")
    # MAKE DATASET
    data = AsistDataset(
        acoustic_dict,
        glove,
        splits=1,
        sequence_prep="pad",
        truncate_from="start",
        norm=None,
        add_avging=params.model_params.add_avging,
        transcript_type="zoom",
    )

    # get data for testing
    test_data = data.current_split
    test_ds = DatumListDataset(test_data, None)

    # predict
    ordered_predictions, ordered_penult_lyrs = multitask_predict_without_gold_labels(
        trained_model,
        test_ds,
        params.model_params.batch_size,
        device,
        num_predictions=2,
        avgd_acoustic=avgd_acoustic,
        use_speaker=params.model_params.use_speaker,
        use_gender=params.model_params.use_gender,
        get_prob_dist=True,
        return_penultimate_layer=True,
    )

    # get the df to save in new json
    df = printdict([], acoustic_dict, ordered_predictions, ordered_penult_lyrs)

    # create and save new json
    prediction_json = create_new_json(df)
    return prediction_json


def get_json_output_of_speech_analysis(
    list_of_json_objs, trained_model_file, glove, params
):
    """
    Use this function to take a list of json input objects
    and a trained model
    and return a new json with metadata and model predictions
    :param list_of_json_objs: a list of json objects
    :param trained_model_file: a trained pytorch model
    :param glove: an instance of Glove
    :param params: the parameters used with the trained model
    """
    # # Set device, checking CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CREATE NN
    # get set of pretrained embeddings and their shape
    pretrained_embeddings = glove.data
    num_embeddings = pretrained_embeddings.size()[0]
    print(f"shape of pretrained embeddings is: {glove.data.size()}")

    # create test model
    classifier = MultitaskModel(
        params=params.model_params,
        num_embeddings=num_embeddings,
        pretrained_embeddings=pretrained_embeddings,
        multi_dataset=False,
    )
    # get saved parameters
    classifier.load_state_dict(torch.load(trained_model_file))
    classifier.to(device)

    prediction_json = predict_with_model(
        list_of_json_objs, classifier, glove, device, params
    )
    return prediction_json


def save_json_predictions(prediction_json_list, savepath):
    # save the prediction json objects to a single file
    with open(savepath, "w") as sfile:
        for item in prediction_json_list:
            sfile.write(json.dumps(item) + "\n")


def get_data_from_json(list_of_json_objs, avgd_acoustic):
    """
    Take a list of json objects and return a prepared dict
    corresponding to those objects that is formatted for use
    in asist dataset creation
    """
    # instantiate acoustic_dict
    acoustic_dict = {}

    # for file in files (in case more than one)
    for utt_json in list_of_json_objs:
        # get the text, speaker id, utt id, and start time
        utterance = utt_json["data"]["text"]
        speaker = utt_json["data"]["id"]
        utt_id = utt_json["data"]["utterance_id"]
        start_time = utt_json["data"]["word_messages"][0]["start_time"]

        # create features var
        utt_feats = None

        # iterate through utterance, get acoustic feats
        for wd_message in utt_json["data"]["word_messages"]:
            # todo: if we ever want to split by word (e.g. for wd-level RNN)
            #   add that possibility here
            try:
                if utt_feats is None:
                    utt_feats = pd.DataFrame.from_dict(eval(wd_message["features"]))
                else:
                    # get the acoustic info
                    item_feats = pd.DataFrame.from_dict(eval(wd_message["features"]))
                    utt_feats = pd.concat([utt_feats, item_feats], ignore_index=True)
            # sometimes the features data is null
            # in these cases, just skip this data
            except NameError:
                continue
        # average acoustic feats, if required
        if avgd_acoustic:
            utt_feats = pd.DataFrame(utt_feats.mean(axis=0)).transpose()

        # add acoustic data to acoustic dict
        acoustic_dict[utt_id] = utt_feats
        # add necessary columns
        acoustic_dict[utt_id]["speaker"] = speaker
        acoustic_dict[utt_id]["utt"] = utterance
        acoustic_dict[utt_id]["timestart"] = start_time
        # add metadata
        acoustic_dict[utt_id] = add_metadata_to_df(
            acoustic_dict[utt_id], utt_json["msg"]
        )

    return acoustic_dict


def add_metadata_to_df(df, metadata_dict):
    # add necessary metadata to existing df
    df["experiment_id"] = metadata_dict["experiment_id"]
    df["source"] = metadata_dict["source"]
    df["sub_type"] = metadata_dict["sub_type"]
    df["trial_id"] = metadata_dict["trial_id"]
    df["timestamp"] = metadata_dict["timestamp"]
    df["version"] = metadata_dict["version"]

    return df


def create_new_json(pandas_df):
    # create a new json formatted for output

    # get the predictions formatted json
    new_json_list = []

    for index, row in pandas_df.iterrows():
        output_dict = {
            "header": {
                "timestamp": row["timestamp"],
                "message_type": "event",
                "version": row["version"],
            },
            "msg": {
                "experiment_id": row["trial_id"],
                "source": row["source"],
                "sub_type": row["sub_type"],
                "timestamp": row["timestamp"],
                "version": row["version"],
                "filename": row["filename"],
            },
            "data": {
                "speaker": row["speaker"],
                "utterance": row["utt"],
                "start_time": row["timestart"],
                "emotions": {
                    "anger": row["emotion_anger"],
                    "disgust": row["emotion_disgust"],
                    "fear": row["emotion_fear"],
                    "joy": row["emotion_joy"],
                    "neutral": row["emotion_neutral"],
                    "sadness": row["emotion_sadness"],
                    "surprise": row["emotion_surprise"],
                },
                "traits": {
                    "extroversion": row["trait_extroversion"],
                    "neuroticism": row["trait_neuroticism"],
                    "agreeableness": row["trait_agreeableness"],
                    "openness": row["trait_openness"],
                    "conscientiousness": row["trait_conscientiousness"],
                },
                "penultimate_layer_emotions": row["penultimate_layer_emotion"],
                "penultimate_layer_traits": row["penultimate_layer_trait"],
            },
        }
        new_json_list.append(output_dict)

    return new_json_list


def decode_stacked(document, pos=0, decoder=json.JSONDecoder()):
    """
    Decode stacked json messages within a single file
    Adapted from: https://stackoverflow.com/questions/27907633/how-to-extract-multiple-json-objects-from-one-file/50384432
    """
    start_of_json = re.compile(r"[\S]")
    doc = document.read()
    while True:
        match = start_of_json.search(doc, pos)
        if not match:
            return
        pos = match.start()

        try:
            obj, pos = decoder.raw_decode(doc, pos)
        except json.JSONDecodeError as e:
            ex = e
            break
        else:
            ex = None
            yield obj

    if ex is not None:
        raise ex


def printdict(nm, directory, predictions, penult_layers):
    # massage data for use in output json

    # prepare empty df
    df1 = pd.DataFrame()

    # go through all utterance items
    for item in directory:
        # get name of file
        file = [str(item)]
        nm.extend(file)
        # access that file's df
        y = directory[item]  # this is a pandas dataframe

        # keep relevant columns
        sub = y[
            [
                "speaker",
                "utt",
                "timestart",
                "version",
                "trial_id",
                "source",
                "sub_type",
                "timestamp",
            ]
        ].copy(deep=False)
        sub.insert(0, "filename", str(item))
        df1 = pd.concat([df1, sub])

    # add columns with predictions to dataframe:
    df1["emotion_anger"] = [i[0] for i in predictions[0][0]]
    df1["emotion_disgust"] = [i[1] for i in predictions[0][0]]
    df1["emotion_fear"] = [i[2] for i in predictions[0][0]]
    df1["emotion_joy"] = [i[3] for i in predictions[0][0]]
    df1["emotion_neutral"] = [i[4] for i in predictions[0][0]]
    df1["emotion_sadness"] = [i[5] for i in predictions[0][0]]
    df1["emotion_surprise"] = [i[6] for i in predictions[0][0]]

    df1["penultimate_layer_emotion"] = [i for i in penult_layers[0]][0]

    df1["trait_extroversion"] = [i[0] for i in predictions[1][0]]
    df1["trait_neuroticism"] = [i[1] for i in predictions[1][0]]
    df1["trait_agreeableness"] = [i[2] for i in predictions[1][0]]
    df1["trait_openness"] = [i[3] for i in predictions[1][0]]
    df1["trait_conscientiousness"] = [i[4] for i in predictions[1][0]]

    df1["penultimate_layer_trait"] = [i for i in penult_layers[1]][0]

    return df1
