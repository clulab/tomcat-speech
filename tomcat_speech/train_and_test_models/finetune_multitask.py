# train the models created in models directory with MUStARD data
# currently the main entry point into the system
import pickle
import shutil
import sys
import os

#sys.path.append("/home/u18/jmculnan/github/tomcat-speech")
import torch
import numpy as np
import torch.nn as nn
from datetime import date
import random

# from tomcat_speech.data_prep.samplers import RandomOversampler
from tomcat_speech.models.train_and_test_models import make_train_state, \
    train_and_predict_multitask_singledataset
from tomcat_speech.models.multimodal_models import MultitaskModel
from tomcat_speech.models.plot_training import *
from tomcat_speech.train_and_test_models.train_and_test_utils import set_cuda_and_seeds, select_model
from tomcat_speech.train_and_test_models.train_multitask import load_modality_data
# import MultitaskObject and Glove from preprocessing code
sys.path.append("/home/jculnan/github/multimodal_data_preprocessing")
from utils.data_prep_helpers import MultitaskObject, Glove, make_glove_dict #, convert_to_ternary

from tomcat_speech.data_prep.ingest_data import *


def finetune_multitask(all_data_list, loss_fx, sampler, device, output_path, config,
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
    multitask_model = select_model(model_params, num_embeddings, pretrained_embeddings, multidataset=False)

    # load saved model
    saved = torch.load(config.saved_model, map_location=device)
    multitask_model.load_state_dict(saved["model_state_dict"])

    optimizer = torch.optim.Adam(
        lr=model_params.lr,
        params=multitask_model.parameters(),
        weight_decay=model_params.weight_decay,
    )
    optimizer.load_state_dict(saved["optimizer_state_dict"])

    # set the classifier(s) to the right device
    multitask_model = multitask_model.to(device)
    print(multitask_model)

    # create a a save path and file for the model
    model_save_file = f"{item_output_path}/{config.EXPERIMENT_DESCRIPTION}.pt"

    # make the train state to keep track of model training/development
    train_state = make_train_state(model_params.lr, model_save_file,
                                   model_params.early_stopping_criterion)

    # train the model and evaluate on development set
    train_and_predict_multitask_singledataset(
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
        loss_fx=loss_fx,
        use_spec=model_params.use_spec
    )

    # plot the loss and accuracy curves
    # set plot titles
    loss_title = f"Training and Dev loss for model {model_params.model} with lr {model_params.lr}"
    loss_save = f"{item_output_path}/loss.png"
    # plot the loss from model
    plot_train_dev_curve(
        train_vals=train_state["train_loss"],
        dev_vals=train_state["val_loss"],
        x_label="Epoch",
        y_label="Loss",
        title=loss_title,
        save_name=loss_save
    )

    # plot the avg f1 curves for each dataset
    for item in train_state["tasks"]:
        plot_train_dev_curve(
            train_vals=train_state["train_avg_f1"][item],
            dev_vals=train_state["val_avg_f1"][item],
            y_label="Weighted AVG F1",
            title=f"Average f-scores for task {item} for model {model_params.model} with lr {model_params.lr}",
            save_name=f"{item_output_path}/avg-f1_task-{item}.png",
        )

    return train_state["val_best_f1"]


if __name__ == "__main__":
    # import parameters for model
    import tomcat_speech.models.parameters.multitask_config as config

    device = set_cuda_and_seeds(config)

    if not config.model_params.use_distilbert:
        data, loss_fx, sampler, num_embeddings, pretrained_embeddings = load_modality_data(device, config,
                                                                                           use_text=True,
                                                                                           use_acoustic=True)
    else:
        data, loss_fx, sampler = load_modality_data(device, config, use_text=True, use_acoustic=True)
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

            finetune_multitask(data, loss_fx, sampler, device, output_path, config, num_embeddings, pretrained_embeddings)