# train the models created in models directory with MUStARD data
# currently the main entry point into the system
import datetime
import pickle
import shutil
import sys
import os
import time

import pandas as pd
import torch
import numpy as np
from datetime import date
import random

from tomcat_speech.models.train_and_test_models import (
    make_train_state,
    multitask_predict_without_gold_labels,
)
from tomcat_speech.models.multimodal_models import MultitaskModel

# import MultitaskObject and Glove from preprocessing code
sys.path.append("/home/jculnan/github/multimodal_data_preprocessing")
from utils.data_prep_helpers import (
    MultitaskObject,
    Glove,
    make_glove_dict,
    MultitaskTestObject,
)
from tomcat_speech.train_and_test_models.train_multitask import load_modality_data

# set device
cuda = False
if torch.cuda.is_available():
    cuda = True

device = torch.device("cuda" if cuda else "cpu")

# # Check CUDA
if torch.cuda.is_available():
    torch.cuda.set_device(2)


if __name__ == "__main__":
    # set saved model
    saved_model = sys.argv[1]
    # set place to save results of evaluation
    item_output_path = os.path.dirname(saved_model)

    # copy the config file into the experiment directory
    # the second argument should be the config file
    config_path = sys.argv[2]
    config_path = os.path.abspath(config_path)
    output_path = os.path.dirname(config_path)

    shutil.copyfile(
        config_path, "tomcat_speech/train_and_test_models/testing_parameters/config.py"
    )

    # import parameters for model
    from tomcat_speech.train_and_test_models.testing_parameters import config

    # set random seed
    torch.manual_seed(config.model_params.seed)
    np.random.seed(config.model_params.seed)
    random.seed(config.model_params.seed)
    if cuda:
        torch.cuda.manual_seed_all(config.model_params.seed)

    # check if cuda
    print(cuda)

    if cuda:
        # check which GPU used
        print(torch.cuda.current_device())

    # decide if you want to use avgd feats
    avgd_acoustic_in_network = (
        config.model_params.avgd_acoustic or config.model_params.add_avging
    )

    # set location for pickled data (saving or loading)
    if config.USE_SERVER:
        data = "/data/nlp/corpora/MM/pickled_data"
    else:
        data = "data"

    # add stdout to a log file
    with open(os.path.join(output_path, "log"), "a") as f:
        if not config.DEBUG:
            # sys.stdout = f

            # separate from training/dev code
            print("\n")
            print("BEGINNING EVALUATION OF TRAINED MODEL")

            # 1. Load datasets + glove object
            load_path = config.load_path
            feature_set = config.feature_set

            # import data
            #used_ds = pickle.load(open(f"{load_path}", "rb"))
            #print("Data loaded")

            # if data is field-separated
            used_ds = load_modality_data(device, config,
                                         use_text=True,
                                         use_acoustic=True,
                                         use_spectrograms=config.model_params.use_spec)
            #used_ds = used_ds[0][0].train
            all_ds = []
            all_ds.extend(used_ds[0][0].train)
            all_ds.extend(used_ds[0][0].dev)
            all_ds.extend(used_ds[0][0].test)
            used_ds = all_ds
            print("Data loaded")

            # todo: once you incorporate spectrograms
            #   remove this
            #for item in used_ds:
            #    del item['x_spec']

            # if not using distilbert embeddings
            if not config.model_params.use_distilbert:
                # make glove
                glove_dict = make_glove_dict(config.glove_path)
                glove = Glove(glove_dict)

                # get set of pretrained embeddings and their shape
                pretrained_embeddings = glove.data
                num_embeddings = pretrained_embeddings.size()[0]
                print(f"shape of pretrained embeddings is: {glove.data.size()}")

            # this uses train-dev-test folds
            # create instance of model
            if config.model_params.use_distilbert:
                multitask_model = MultitaskModel(
                    params=config.model_params,
                    use_distilbert=config.model_params.use_distilbert,
                    multi_dataset=False
                )
            else:
                multitask_model = MultitaskModel(
                    params=config.model_params,
                    use_distilbert=config.model_params.use_distilbert,
                    num_embeddings=num_embeddings,
                    pretrained_embeddings=pretrained_embeddings,
                    multi_dataset=False
                )

            optimizer = torch.optim.Adam(
                lr=config.model_params.lr,
                params=multitask_model.parameters(),
                weight_decay=config.model_params.weight_decay,
            )

            # set the classifier(s) to the right device
            # get saved parameters
            multitask_model.load_state_dict(torch.load(saved_model, map_location=device))
            multitask_model.to(device)

            print("Trained model loaded, loss function, and optimization prepared")

            # todo: set data sampler?
            sampler = None

            # make the train state to keep track of model training/development
            train_state = make_train_state(
                config.model_params.lr,
                "delete.pt",
                config.model_params.early_stopping_criterion,
            )

            # get order of predictions
            #ordered_ids = [item['audio_id'] for item in used_ds[0]]

            # get predictions
            ordered_predictions, ordered_penult_lyrs, ordered_ids = multitask_predict_without_gold_labels(
                multitask_model,
                used_ds,
                config.model_params.batch_size,
                device,
                num_predictions=3,
                avgd_acoustic=avgd_acoustic_in_network,
                use_speaker=config.model_params.use_speaker,
                use_gender=config.model_params.use_gender,
                get_prob_dist=True,
                return_penultimate_layer=True,
                use_spec=config.model_params.use_spec,
            )

            all_preds = {}
            for pred_type in ordered_predictions.keys():
                all_preds[pred_type] = []
                #print(pred_type)
                #print("--------------------")
                for group_pred in ordered_predictions[pred_type]:
                    for preds in group_pred:
                        all_preds[pred_type].append(preds.index(max(preds)))
                        print(preds.index(max(preds)))

            data_to_save = pd.DataFrame(all_preds)
            data_to_save.columns = ['trait', 'emotion', 'sentiment']
            data_to_save['audio_id'] = ordered_ids
            #data_to_save.to_csv(f"../../PROJECTS/ToMCAT/Evaluating_modelpredictions/data_from_speechAnalyzer/used_for_evaluating_model_results/all_preds_no_classweights.csv", index=False)
            data_to_save.to_csv(f"/media/jculnan/backup/jculnan/datasets/asist_data/mmc_hiercnn_spec_glove_preds.csv", index=False)