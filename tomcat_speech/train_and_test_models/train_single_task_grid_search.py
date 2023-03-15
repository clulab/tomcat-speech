import pickle

import shutil
import sys
import os
import torch
import numpy as np
from datetime import date
import random

import sys
sys.path.append("/home/cheonkamjeong/multimodal_data_preprocessing")
from tomcat_speech.models.train_and_test_models import train_and_predict, make_train_state
from tomcat_speech.models.multimodal_models import MultitaskModel
from tomcat_speech.train_and_test_models.train_multitask import load_modality_data
from tomcat_speech.models.plot_training import *
from tomcat_speech.train_and_test_models.train_and_test_utils import set_cuda_and_seeds, select_model

# import MultitaskObject and Glove from preprocessing code
sys.path.append("../multimodal_data_preprocessing")
from utils.data_prep_helpers import MultitaskObject, Glove, make_glove_dict


def train_single_task(all_data_list, loss_fx, sampler, device, output_path, config,
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
    task_model = select_model(model_params, num_embeddings, pretrained_embeddings)

    optimizer = torch.optim.Adam(
        lr=model_params.lr,
        params=task_model.parameters(),
        weight_decay=model_params.weight_decay,
    )

    # set the classifier(s) to the right device
    multitask_model = task_model.to(device)
    print(multitask_model)

    # create a save path and file for the model
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
    import tomcat_speech.models.parameters.singletask_config as config

    # set cuda and random seeds
    device = set_cuda_and_seeds(config)

    # get data
    if not config.model_params.use_distilbert:
        data, loss_fx, sampler, num_embeddings, pretrained_embeddings = load_modality_data(device, config, use_acoustic=False)
    else:
        data, loss_fx, sampler = load_modality_data(device, config, use_acoustic=False)
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



    '''
    문제가 하나 있는데, 함수 맨 앞에 config가 아닌 다른 인자가 들어가면 작동을 하지 않습니다.

    해결방법이 있을 것 같긴 한데, 우선 임시방편으로 맨 앞에 config 인자는 고정으로 들어가야 하는 것 같습니다.
    그래서 train_single_task 함수 자체를

    def train_single_task(config, data, loss_fx, ...)

    위처럼 config를 맨 앞 argument로 사용해서 함수 사용하시면 될 것 같고(같은 내용의 함수를 argument 순서만 바꾸시면 될 것 같아요.)

    config.lr = tune.grid_search([1e-3, 1e-4, 1e-5])
    config.dropout = tune.grid_search([0.2, 0.3, 0.4])
    config.batch_size = tune.grid_search([16, 32, 64, 128])

    result = tune.run(
        tune.with_parameters(train_with_single_task, data=data, loss_fx=loss_fx, ...),
        config=config
    )

    위처럼 train-with_single_task의 config를 제외한 나머지 argument들을 tune.with_paramters() 안에 집어넣고
    두번째 인자로 config=config
    이렇게 넣어서 돌리시면 될 것 같습니다.

    결론적으로 아래 코드를

    with open(os.path.join(output_path, "log"), "a") as f:
        if not config.DEBUG:
            sys.stdout = f

            result = tune.run(
                tune.with_parameters(data=data, loss_fx=loss_fx, sampler=sampler, device=device, \
                    output_path=output_path, num_embeddings=num_embeddings, pretrained_embeddings=pretrained_embeddings)
                ),
                config=config
            )
    
    위처럼 바꾸시면 될 거 같습니다.

    혹시 돌려보시고 오류 나시면 연락주세요.
    '''
    # add stdout to a log file
    with open(os.path.join(output_path, "log"), "a") as f:
        if not config.DEBUG:
            sys.stdout = f

            train_single_task(data, loss_fx, sampler, device, output_path, config, num_embeddings, pretrained_embeddings)
