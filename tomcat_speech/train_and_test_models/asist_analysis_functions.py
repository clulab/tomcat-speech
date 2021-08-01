import random
import json
import re
import numpy as np
import torch
import sys
from tomcat_speech.data_prep.asist_data.asist_dataset_creation import AsistDataset
from tomcat_speech.models.train_and_test_models import (
    multitask_predict_without_gold_labels,
)
from tomcat_speech.models.input_models import MultitaskModel

from tomcat_speech.data_prep.data_prep_helpers import (
    make_glove_dict,
    Glove,
    DatumListDataset,
)

# Import parameters for model
# import tomcat_speech.train_and_test_models.testing_parameters.config as params
import pandas as pd


def get_json_output_asist_analysis(
    input_aligned_json, trained_model, glove_file, output_filepath, params
):
    """
    Use this function to take a json input and a trained model
    and return a new json with metadata and model predictions
    """
    # Set device, checking CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    acoustic_dict, utts_dict = get_data_from_json(input_aligned_json, avgd_acoustic)
    print("Acoustic dict created")

    # IMPORT GLOVE + MAKE GLOVE OBJECT
    glove_dict = make_glove_dict(glove_file)
    glove = Glove(glove_dict)
    print("Glove object created")

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

    # CREATE NN
    # get set of pretrained embeddings and their shape
    pretrained_embeddings = glove.data
    num_embeddings = pretrained_embeddings.size()[0]
    print(f"shape of pretrained embeddings is: {data.glove.data.size()}")

    # create test model
    classifier = MultitaskModel(
        params=params.model_params,
        num_embeddings=num_embeddings,
        pretrained_embeddings=pretrained_embeddings,
        multi_dataset=False,
    )
    # get saved parameters
    classifier.load_state_dict(torch.load(trained_model))
    classifier.to(device)

    # test the model
    ordered_predictions, ordered_penult_lyrs = multitask_predict_without_gold_labels(
        classifier,
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
    create_new_json(output_filepath, utts_dict, df)


def get_data_from_json(aligned_json_file, avgd_acoustic):
    # instantiate utts_dict, acoustic_dict
    acoustic_dict = {}
    utts_dict = {}

    # for file in files (in case more than one)
    for json_file in aligned_json_file:

        # open the file with multiple jsons
        with open(json_file, "r") as j_file:
            # get the data from all jsons in this file
            all_jsons = decode_stacked(j_file)

            # for each utterance
            for utt_json in all_jsons:

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
                            utt_feats = pd.DataFrame.from_dict(
                                eval(wd_message["features"])
                            )
                        else:
                            # get the acoustic info
                            item_feats = pd.DataFrame.from_dict(
                                eval(wd_message["features"])
                            )
                            utt_feats = pd.concat(
                                [utt_feats, item_feats], ignore_index=True
                            )
                    # sometimes the features data is null
                    # in these cases, just skip this data
                    except NameError:
                        continue
                # average acoustic feats, if required
                if avgd_acoustic:
                    utt_feats = pd.DataFrame(utt_feats.mean(axis=0)).transpose()

                # add metadata to utts dict
                utts_dict[utt_id] = {
                    "speaker": speaker,
                    "utt": utterance,
                    "timestart": start_time,
                    "msg": utt_json["msg"],
                }

                # add acoustic data to acoustic dict
                acoustic_dict[utt_id] = utt_feats
                acoustic_dict[utt_id]["speaker"] = speaker
                acoustic_dict[utt_id]["utt"] = utterance
                acoustic_dict[utt_id]["timestart"] = start_time

    return acoustic_dict, utts_dict


def create_new_json(output_filepath, utts_dict, pandas_df):
    # Final output:
    with open(output_filepath, "w") as f:
        for index, row in pandas_df.iterrows():
            vers, exp = get_metadata(utts_dict[row["filename"]])
            output_dict = {
                "header": {
                    "timestamp": row["timestart"],
                    # "timestamp": "some_time_stamp",
                    "message_type": "event",
                    "version": vers,
                },
                "msg": {
                    "source": "TomcatSpeechAnalyzer",
                    "experiment_id": exp,
                    "timestamp": row["timestart"],
                    "sub_type": "Event:speech_feature",
                    "version": vers,
                    "filename": row["filename"],
                },
                "data": {
                    "speaker": row["speaker"],
                    "utterance": row["utt"],
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
            f.write(json.dumps(output_dict) + "\n")


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
    df1 = pd.DataFrame()
    for item in directory:
        file = [str(item)]
        nm.extend(file)
        y = directory[item]  # this is a pandas dataframe
        # y["Filename"] = file
        sub = y[["speaker", "utt", "timestart"]].copy(deep=False)
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


def get_metadata(input_json):
    # get the input from utterance json
    # version
    version = input_json["msg"]["version"]
    # experiment id
    exp_id = input_json["msg"]["experiment_id"]

    return version, exp_id


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_filepath",
        help="Path to output file",
        default="output/asist_output.txt",
        nargs="?",
    )
    parser.add_argument(
        "--glove_file",
        help="Path to Glove file",
        default="data/glove.short.300d.punct.txt",
        nargs="?",
    )
    parser.add_argument(
        "--emotion_model",
        help="Path to saved model you would like to use in testing",
        default="data/EMOTION_MODEL_FOR_ASIST_batch100_100hidden_2lyrs_lr0.01.pth",
        nargs="?",
    )
    parser.add_argument(
        "--input_aligned_json",
        help="Input json file(s) to get predictions on",
        nargs="+",
    )

    args = parser.parse_args()

    # the current parameters file is saved as testing_parameters/config.py
    import tomcat_speech.train_and_test_models.testing_parameters.config as params

    get_json_output_asist_analysis(
        args.input_aligned_json,
        args.emotion_model,
        args.glove_file,
        args.output_filepath,
        params,
    )
