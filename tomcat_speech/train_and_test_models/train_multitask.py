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

from tomcat_speech.data_prep.samplers import RandomOversampler
from tomcat_speech.models.train_and_test_models import train_and_predict, make_train_state
from tomcat_speech.models.multimodal_models import MultitaskModel
from tomcat_speech.models.plot_training import *
from tomcat_speech.train_and_test_models.train_and_test_utils import set_cuda_and_seeds, select_model

# import MultitaskObject and Glove from preprocessing code
sys.path.append("../multimodal_data_preprocessing")
from utils.data_prep_helpers import MultitaskObject, Glove, make_glove_dict

from tomcat_speech.data_prep.ingest_data import *


def load_modality_data(device, config):
    """
    Load the modality-separated data
    """
    pass


def load_data(device, config):
    # 1. Load datasets + glove object
    load_path = config.load_path
    feature_set = config.feature_set

    # import cdc data
    # cdc_train_ds = pickle.load(open(f"{load_path}/cdc_{feature_set}_train.pickle", "rb"))
    # cdc_dev_ds = pickle.load(open(f"{load_path}/cdc_{feature_set}_dev.pickle", "rb"))
    # cdc_test_ds = pickle.load(open(f"{load_path}/cdc_{feature_set}_test.pickle", "rb"))
    # cdc_weights = pickle.load(open(f"{load_path}/cdc_{feature_set}_clsswts.pickle", "rb"))
    # print("CDC data loaded")
    #
    # # import cmu mosi data
    # mosi_train_ds = pickle.load(open(f"{load_path}/mosi_{feature_set}_train.pickle", "rb"))
    # mosi_dev_ds = pickle.load(open(f"{load_path}/mosi_{feature_set}_dev.pickle", "rb"))
    # mosi_test_ds = pickle.load(open(f"{load_path}/mosi_{feature_set}_test.pickle", "rb"))
    # mosi_weights = pickle.load(open(f"{load_path}/mosi_{feature_set}_clsswts.pickle", "rb"))
    # print("CMU MOSI data loaded")

    # import firstimpr data
    firstimpr_train_ds = pickle.load(open(f"{load_path}/firstimpr_{feature_set}_train.pickle", "rb"))
    firstimpr_dev_ds = pickle.load(open(f"{load_path}/firstimpr_{feature_set}_dev.pickle", "rb"))
    firstimpr_test_ds = pickle.load(open(f"{load_path}/firstimpr_{feature_set}_test.pickle", "rb"))
    # firstimpr_weights = pickle.load(open(f"{load_path}/firstimpr_{feature_set}_clsswts.pickle", "rb"))
    print("FirstImpr data loaded")

    # import meld data
    meld_train_ds = pickle.load(open(f"{load_path}/meld_{feature_set}_train.pickle", "rb"))
    meld_dev_ds = pickle.load(open(f"{load_path}/meld_{feature_set}_dev.pickle", "rb"))
    meld_test_ds = pickle.load(open(f"{load_path}/meld_{feature_set}_test.pickle", "rb"))
    # meld_weights = pickle.load(open(f"{load_path}/meld_{feature_set}_clsswts.pickle", "rb"))
    print("MELD data loaded")

    # # import ravdess data
    # ravdess_train_ds = pickle.load(open(f"{load_path}/ravdess_{feature_set}_train.pickle", "rb"))
    # ravdess_dev_ds = pickle.load(open(f"{load_path}/ravdess_{feature_set}_dev.pickle", "rb"))
    # ravdess_test_ds = pickle.load(open(f"{load_path}/ravdess_{feature_set}_test.pickle", "rb"))
    # ravdess_weights = pickle.load(open(f"{load_path}/ravdess_{feature_set}_clsswts.pickle", "rb"))
    # print("RAVDESS data loaded")

    # if not using distilbert embeddings
    if not config.model_params.use_distilbert:
        # make glove
        glove_dict = make_glove_dict(config.glove_path)
        glove = Glove(glove_dict)

        # get set of pretrained embeddings and their shape
        pretrained_embeddings = glove.data
        num_embeddings = pretrained_embeddings.size()[0]
        print(f"shape of pretrained embeddings is: {glove.data.size()}")

    # get number of items in all datasets
    # total_data_size = len(cdc_train_ds) + len(mosi_train_ds) + len(firstimpr_train_ds) + \
    #                   len(meld_train_ds) + len(ravdess_train_ds)

    # # add loss function for cdc
    # cdc_loss_func = torch.nn.CrossEntropyLoss(
    #     weight=cdc_weights.to(device),
    #     reduction="mean"
    # )
    #
    # cdc_obj = MultitaskObject(
    #     cdc_train_ds,
    #     cdc_dev_ds,
    #     cdc_test_ds,
    #     cdc_loss_func,
    #     task_num=0
    # )

    # cdc_obj.change_loss_multiplier(2)
    # cdc_obj.change_loss_multiplier(len(cdc_train_ds) / float(total_data_size))

    # # add loss func, multitask obj for cmu mosi
    # mosi_loss_func = torch.nn.CrossEntropyLoss(
    #     weight=mosi_weights.to(device),
    #     reduction="mean"
    # )
    #
    # mosi_obj = MultitaskObject(
    #     mosi_train_ds,
    #     mosi_dev_ds,
    #     mosi_test_ds,
    #     mosi_loss_func,
    #     task_num=1
    # )

    # mosi_obj.change_loss_multiplier(7)
    # mosi_obj.change_loss_multiplier(len(mosi_train_ds) / float(total_data_size))

    # add loss function for firstimpr
    firstimpr_loss_func = torch.nn.CrossEntropyLoss(
        # weight=firstimpr_weights.to(device),
        reduction="mean"
    )
    # create multitask object
    firstimpr_obj = MultitaskObject(
        firstimpr_train_ds,
        firstimpr_dev_ds,
        firstimpr_test_ds,
        firstimpr_loss_func,
        task_num=1,
    )

    # firstimpr_obj.change_loss_multiplier(5)
    # firstimpr_obj.change_loss_multiplier(len(firstimpr_train_ds) / float(total_data_size))

    # # add loss function for meld
    meld_loss_func = torch.nn.CrossEntropyLoss(
        # weight=meld_weights.to(device),
        reduction="mean"
    )
    # create multitask object
    meld_obj = MultitaskObject(
        meld_train_ds,
        meld_dev_ds,
        meld_test_ds,
        meld_loss_func,
        task_num=0,
    )

    # meld_obj.change_loss_multiplier(7)
    # meld_obj.change_loss_multiplier(len(meld_train_ds) / float(total_data_size))

    # # add loss function, multitask obj for ravdess
    # ravdess_loss_func = torch.nn.CrossEntropyLoss(
    #     weight=ravdess_weights.to(device),
    #     reduction="mean"
    # )
    #
    # ravdess_obj = MultitaskObject(
    #     ravdess_train_ds,
    #     ravdess_dev_ds,
    #     ravdess_test_ds,
    #     ravdess_loss_func,
    #     task_num=4
    # )

    # ravdess_obj.change_loss_multiplier(2)
    # ravdess_obj.change_loss_multiplier(len(ravdess_train_ds) / float(total_data_size))

    # set all data list
    all_data_list = [
        # cdc_obj,
        # mosi_obj,
        meld_obj,
        firstimpr_obj,
        # meld_obj,
        # ravdess_obj
    ]

    # create a single loss function
    if config.model_params.single_loss:
        loss_fx = torch.nn.CrossEntropyLoss(reduction="mean")
    else:
        loss_fx = None

    print(
        "Model, loss function, and optimization created"
    )

    # sampler = None
    if config.model_params.use_sampler:
        sampler = RandomOversampler(config.model_params.seed)
    else:
        sampler = None
    # sampler = BatchSchedulerSampler()

    if not config.model_params.use_distilbert:
        return all_data_list, loss_fx, sampler, num_embeddings, pretrained_embeddings
    else:
        return all_data_list, loss_fx, sampler


def train_multitask(all_data_list, loss_fx, sampler, device, output_path, config,
                    num_embeddings=None, pretrained_embeddings=None, extra_params=None):
    if extra_params:
        model_params = extra_params
    else:
        model_params = config.model_params

    # decide if you want to use avgd feats
    avgd_acoustic_in_network = (
        model_params.avgd_acoustic or model_params.add_avging
    )

    # 3. CREATE NN
    print(model_params)

    item_output_path = os.path.join(
        output_path,
        f"LR{model_params.lr}_BATCH{model_params.batch_size}_"
        f"NUMLYR{model_params.num_gru_layers}_"
        f"SHORTEMB{model_params.short_emb_dim}_"
        f"INT-OUTPUT{model_params.output_dim}_"
        f"DROPOUT{model_params.dropout}_"
        f"FC-FINALDIM{model_params.final_hidden_dim}",
    )

    # make sure the full save path exists; if not, create it
    os.system(
        'if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(
            item_output_path
        )
    )


    # this uses train-dev-test folds
    # create instance of model
    multitask_model = select_model(model_params, num_embeddings, pretrained_embeddings)

    optimizer = torch.optim.Adam(
        lr=model_params.lr,
        params=multitask_model.parameters(),
        weight_decay=model_params.weight_decay,
    )

    # set the classifier(s) to the right device
    multitask_model = multitask_model.to(device)
    print(multitask_model)

    # create a a save path and file for the model
    model_save_file = f"{item_output_path}/{config.EXPERIMENT_DESCRIPTION}.pt"

    # make the train state to keep track of model training/development
    train_state = make_train_state(model_params.lr, model_save_file,
                                   model_params.early_stopping_criterion)

    # train the model and evaluate on development set
    train_and_predict(
        multitask_model,
        train_state,
        all_data_list,
        model_params.batch_size,
        model_params.num_epochs,
        optimizer,
        device,
        scheduler=None,
        sampler=sampler,
        avgd_acoustic=avgd_acoustic_in_network,
        use_speaker=model_params.use_speaker,
        use_gender=model_params.use_gender,
        loss_fx=loss_fx
    )

    # plot the loss and accuracy curves
    # set plot titles
    loss_title = f"Training and Dev loss for model {model_params.model} with lr {model_params.lr}"
    loss_save = f"{item_output_path}/loss.png"
    # plot the loss from model
    plot_train_dev_curve(
        train_state["train_loss"],
        train_state["val_loss"],
        x_label="Epoch",
        y_label="Loss",
        title=loss_title,
        save_name=loss_save
    )

    # plot the avg f1 curves for each dataset
    for item in train_state["tasks"]:
        plot_train_dev_curve(
            train_state["train_avg_f1"][item],
            train_state["val_avg_f1"][item],
            x_label="Epoch",
            y_label="Weighted AVG F1",
            title=f"Average f-scores for task {item} for model {model_params.model} with lr {model_params.lr}",
            save_name=f"{item_output_path}/avg-f1_task-{item}.png",
        )


if __name__ == "__main__":
    # import parameters for model
    import tomcat_speech.models.parameters.multitask_config as config

    device = set_cuda_and_seeds(config)

    if not config.model_params.use_distilbert:
        data, loss_fx, sampler, num_embeddings, pretrained_embeddings = load_data(device, config)
    else:
        data, loss_fx, sampler = load_data(device, config)
        num_embeddings = None
        pretrained_embeddings = None

    # create save location
    output_path = os.path.join(
        config.exp_save_path,
        str(config.EXPERIMENT_ID)
        + "_"
        + config.EXPERIMENT_DESCRIPTION
        + str(date.today()),
    )

    print(f"OUTPUT PATH:\n{output_path}")

    # make sure the full save path exists; if not, create it
    os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(output_path))

    # copy the config file into the experiment directory
    shutil.copyfile(config.CONFIG_FILE, os.path.join(output_path, "config.py"))

    # add stdout to a log file
    with open(os.path.join(output_path, "log"), "a") as f:
        if not config.DEBUG:
            sys.stdout = f

            train_multitask(data, loss_fx, sampler, device, output_path, config, num_embeddings, pretrained_embeddings)
