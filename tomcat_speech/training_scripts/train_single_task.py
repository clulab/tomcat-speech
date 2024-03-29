# train a single-task model for any of the tasks
# combining the training of the separate train_task models into one
# Cheonkam has started alterations to this file

import shutil
import sys
import os

import torch
from datetime import date

from tomcat_speech.training_and_evaluation_functions.train_and_test_single_models import (
    train_and_predict,
    make_train_state,
)
from tomcat_speech.training_and_evaluation_functions.plot_training import *
from tomcat_speech.training_and_evaluation_functions.train_and_test_utils import (
    set_cuda_and_seeds,
    select_model,
)
from tomcat_speech.training_and_evaluation_functions.loading_data import (
    load_modality_data,
)


def train_single_task(
    all_data_list,
    loss_fx,
    sampler,
    device,
    output_path,
    config,
    num_embeddings=None,
    pretrained_embeddings=None,
    extra_params=None,
    pretrained_model=None,
):
    if extra_params:
        model_params = extra_params
    else:
        model_params = config.model_params

    # decide if you want to use avgd feats
    avgd_acoustic_in_network = model_params.avgd_acoustic or model_params.add_avging

    # CREATE NN
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
    os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(item_output_path))

    # this uses train-dev-test folds
    # create instance of model
    task_model = select_model(model_params, num_embeddings, pretrained_embeddings)

    optimizer = torch.optim.Adam(
        lr=model_params.lr,
        params=task_model.parameters(),
        weight_decay=model_params.weight_decay,
    )

    # if we are loading pretrained files for fine-tuning, add these here
    if pretrained_model is not None:
        task_model.load_state_dict(pretrained_model["model_state_dict"])
        optimizer.load_state_dict(pretrained_model["optimizer_state_dict"])

    # set the classifier(s) to the right device
    multitask_model = task_model.to(device)
    print(multitask_model)

    # create a save path and file for the model
    model_save_file = f"{item_output_path}/{config.EXPERIMENT_DESCRIPTION}.pt"

    # make the train state to keep track of model training/development
    train_state = make_train_state(
        model_params.lr, model_save_file, model_params.early_stopping_criterion
    )

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
        # use_spec = False #added 10/24/22
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
        save_name=loss_save,
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

    return train_state["val_best_f1"]


if __name__ == "__main__":
    # import parameters for model
    import tomcat_speech.parameters.singletask_config as config

    # set cuda and random seeds
    device = set_cuda_and_seeds(config)

    # get data
    if not config.model_params.use_distilbert:
        (
            data,
            loss_fx,
            sampler,
            num_embeddings,
            pretrained_embeddings,
        ) = load_modality_data(device, config)
    else:
        data, loss_fx, sampler = load_modality_data(device, config)
        num_embeddings = None
        pretrained_embeddings = None

    print(type(data[0].train))  # 12/15/22
    print(len(data[0].train))  # 12/15/22

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

    # get pretrained model if fine-tuning
    if config.pretrained_model is not None:
        pretrained = torch.load(config.pretrained_model, map_location=device)
    else:
        pretrained = None

    # add stdout to a log file
    with open(os.path.join(output_path, "log"), "a") as f:
        if not config.DEBUG:
            sys.stdout = f

            train_single_task(
                data,
                loss_fx,
                sampler,
                device,
                output_path,
                config,
                num_embeddings,
                pretrained_embeddings,
                pretrained_model=pretrained,
            )
