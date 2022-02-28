# test a single-task model for any of the five tasks
# combining the testing of the separate train_task models into one

# train the models created in models directory with MUStARD data
# currently the main entry point into the system
import pickle
import shutil
import sys
import os
import torch
import numpy as np
from datetime import date
import random

from tomcat_speech.models.train_and_test_models import (
    evaluate,
    train_and_predict,
    make_train_state,
)
from tomcat_speech.models.multimodal_models import MultitaskModel
from tomcat_speech.models.plot_training import *

# import MultitaskObject and Glove from preprocessing code
sys.path.append("../multimodal_data_preprocessing")
from utils.data_prep_helpers import (
    MultitaskObject,
    Glove,
    make_glove_dict,
    MultitaskTestObject,
)

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

    shutil.copyfile(config_path, "mmml/config_files/test_config.py")

    # import parameters for model
    import mmml.config_files.test_config as config

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
            sys.stdout = f

            # 1. Load datasets + glove object
            load_path = config.load_path
            feature_set = config.feature_set

            # get
            test_ds = pickle.load(
                open(f"{load_path}/{config.task}_{feature_set}_test.pickle", "rb")
            )
            class_weights = pickle.load(
                open(f"{load_path}/{config.task}_{feature_set}_clsswts.pickle", "rb")
            )

            # if not using distilbert embeddings
            if not config.model_params.use_distilbert:
                # make glove
                glove_dict = make_glove_dict(config.glove_path)
                glove = Glove(glove_dict)

                # get set of pretrained embeddings and their shape
                pretrained_embeddings = glove.data
                num_embeddings = pretrained_embeddings.size()[0]
                print(f"shape of pretrained embeddings is: {glove.data.size()}")

            # 3. CREATE NN
            # this uses train-dev-test folds
            # create instance of model
            if config.model_params.use_distilbert:
                task_model = MultitaskModel(
                    params=config.model_params,
                    use_distilbert=config.model_params.use_distilbert,
                )
            else:
                task_model = MultitaskModel(
                    params=config.model_params,
                    use_distilbert=config.model_params.use_distilbert,
                    num_embeddings=num_embeddings,
                    pretrained_embeddings=pretrained_embeddings,
                )

            optimizer = torch.optim.Adam(
                lr=config.model_params.lr,
                params=task_model.parameters(),
                weight_decay=config.model_params.weight_decay,
            )

            # set the classifier(s) to the right device
            # get saved parameters
            task_model.load_state_dict(torch.load(saved_model, map_location=device))
            task_model.to(device)

            # add loss function for cdc
            loss_func = torch.nn.CrossEntropyLoss(
                weight=class_weights.to(device), reduction="mean"
            )

            task_obj = MultitaskTestObject(test_ds, loss_func, task_num=0)

            # set all data list
            all_data_list = [task_obj]

            print("Trained model loaded, loss function, and optimization prepared")

            # todo: set data sampler?
            sampler = None

            # make the train state to keep track of model training/development
            train_state = make_train_state(
                config.model_params.lr,
                "delete.pt",
                config.model_params.early_stopping_criterion,
            )

            # train the model and evaluate on development set
            evaluate(
                task_model,
                train_state,
                all_data_list,
                config.model_params.batch_size,
                pickle_save_name=f"{item_output_path}/test_results.pickle",
                device=device,
                avgd_acoustic=avgd_acoustic_in_network,
                use_speaker=config.model_params.use_speaker,
                use_gender=config.model_params.use_gender,
            )
