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
from tomcat_speech.models.train_and_test_models import train_and_predict, make_train_state
from tomcat_speech.models.multimodal_models import MultitaskModel
from tomcat_speech.models.plot_training import *
from tomcat_speech.train_and_test_models.train_and_test_utils import set_cuda_and_seeds, select_model

# import MultitaskObject and Glove from preprocessing code
sys.path.append("/home/jculnan/github/multimodal_data_preprocessing")
from utils.data_prep_helpers import MultitaskObject, Glove, make_glove_dict #, convert_to_ternary

from tomcat_speech.data_prep.ingest_data import *


def load_modality_data(device, config, use_text=True, use_acoustic=True):
    """
    Load the modality-separated data
    """
    use_spectrograms = config.model_params.use_spec
    load_path = config.load_path
    feature_set = config.feature_set
    feature_type, embedding_type = feature_set.split("_")[:2]

    all_datasets = {}
    task_num = 0
    total_data_size = 0

    # iterate through datasets listed in config file
    for dataset in config.datasets:
        dataset = dataset.lower()
        print(f"Now loading dataset {dataset}")
        train_modalities = []
        dev_modalities = []
        test_modalities = []
        if use_text:
            print(f"Loading Text Features for embedding type {embedding_type}")
            text_base = f"{load_path}/field_separated_data/text_data/{embedding_type}/{dataset}_{embedding_type}"
            train_text = pickle.load(open(f"{text_base}_train.pickle", "rb"))
            train_modalities.append(train_text)
            dev_text = pickle.load(open(f"{text_base}_dev.pickle", "rb"))
            dev_modalities.append(dev_text)
            test_text = pickle.load(open(f"{text_base}_test.pickle", "rb"))
            test_modalities.append(test_text)
            del train_text, dev_text, test_text
        if use_acoustic:
            print(f"Loading Acoustic Features for acoustic set {feature_type}")
            audio_base = f"{load_path}/field_separated_data/acoustic_data/{feature_type}/{dataset}_{feature_type}"
            train_audio = pickle.load(open(f"{audio_base}_train.pickle", "rb"))
            train_modalities.append(train_audio)
            dev_audio = pickle.load(open(f"{audio_base}_dev.pickle", "rb"))
            dev_modalities.append(dev_audio)
            test_audio = pickle.load(open(f"{audio_base}_test.pickle", "rb"))
            test_modalities.append(test_audio)
            del train_audio, dev_audio, test_audio
        if use_spectrograms:
            print("Loading Spectrogram Features")
            spec_base = f"{load_path}/field_separated_data/spectrogram_data/{dataset}_spec"
            print("Loading spec features for train set")
            train_spec = pickle.load(open(f"{spec_base}_train.pickle", "rb"))
            print("Spec features loaded for train set")

            longest = 300
            x_replacer = torch.zeros(longest, 513)
            for item in train_spec:
                x = x_replacer.detach().clone()
                x[:min(len(item['x_spec']), longest), :513] = item['x_spec'][:min(len(item['x_spec']), longest)]
                item['x_spec'] = x
            train_modalities.append(train_spec)
            del train_spec

            print("Loading spec features for dev set")
            dev_spec = pickle.load(open(f"{spec_base}_dev.pickle", "rb"))
            for item in dev_spec:
                x = x_replacer.detach().clone()
                x[:min(len(item['x_spec']), longest), :513] = item['x_spec'][:min(len(item['x_spec']), longest)]
                item['x_spec'] = x
            dev_modalities.append(dev_spec)
            del dev_spec

            print("loading spec features for test set")
            test_spec = pickle.load(open(f"{spec_base}_test.pickle", "rb"))
            for item in test_spec:
                x = x_replacer.detach().clone()
                x[:min(len(item['x_spec']), longest), :513] = item['x_spec'][:min(len(item['x_spec']), longest)]
                item['x_spec'] = x
            test_modalities.append(test_spec)
            del test_spec

        # add ys data and classweights to this
        print(f"Loading gold labels for dataset {dataset}")
        ys_base = f"{load_path}/field_separated_data/ys_data/{dataset}_ys"
        ys_train = pickle.load(open(f"{ys_base}_train.pickle", "rb"))
        ys_dev = pickle.load(open(f"{ys_base}_dev.pickle", "rb"))
        ys_test = pickle.load(open(f"{ys_base}_test.pickle", "rb"))

        #if dataset == "mosi":
        #    ys_train = convert_to_ternary(ys_train)
        #    ys_dev = convert_to_ternary(ys_dev)
        #    ys_test = convert_to_ternary(ys_test)

        train_modalities.append(ys_train)
        dev_modalities.append(ys_dev)
        test_modalities.append(ys_test)

        del ys_train, ys_dev, ys_test

        # combine modalities
        if len(train_modalities) < 2:
            exit("No modalities data has been loaded; please select at least one modality to load")

        train_data = combine_modality_data(train_modalities)
        dev_data = combine_modality_data(dev_modalities)
        test_data = combine_modality_data(test_modalities)

        del train_modalities, dev_modalities, test_modalities

        clsswts_base = f"{load_path}/field_separated_data/class_weights/{dataset}_clsswts.pickle"
        clsswts = pickle.load(open(clsswts_base, "rb"))

        dset_loss_func = torch.nn.CrossEntropyLoss(
            weight=clsswts.to(device) if config.model_params.use_clsswts else None,
            reduction="mean"
        )

        all_datasets[task_num] = MultitaskObject(train_data,
                                                 dev_data,
                                                 test_data,
                                                 dset_loss_func,
                                                 task_num=task_num)
        # increment task number
        task_num += 1

        # add to the total data size
        total_data_size += len(train_data)

    all_data_list = [all_datasets[item] for item in sorted(all_datasets.keys())]

    del all_datasets

    # optionally change loss multiplier
    if config.model_params.loss_multiplier:
        for obj in all_data_list:
            obj.change_loss_multiplier(len(obj.train_ds) / float(total_data_size))

    # if not using distilbert embeddings
    if not config.model_params.use_distilbert:
        # make glove
        glove_dict = make_glove_dict(config.glove_path)
        glove = Glove(glove_dict)

        # get set of pretrained embeddings and their shape
        pretrained_embeddings = glove.data
        num_embeddings = pretrained_embeddings.size()[0]

    # create a single loss function
    if config.model_params.single_loss:
        loss_fx = torch.nn.CrossEntropyLoss(reduction="mean")
    else:
        loss_fx = None

    print(
        "Model, loss function, and optimization created"
    )

    # set sampler
    sampler = None
    if not config.model_params.use_distilbert:
        return all_data_list, loss_fx, sampler, num_embeddings, pretrained_embeddings
    else:
        return all_data_list, loss_fx, sampler


def combine_modality_data(list_of_modality_data):
    """
    Use a list of lists of dicts (each of which contains info on a modality)
    to get a single list of dicts for the dataset
    return this single list of dicts
    """
    all_data = {}

    # get all utterance IDs
    for item in list_of_modality_data[0]:
        all_data[item['audio_id']] = item

    for dataset in list_of_modality_data[1:]:
        for item in dataset:
            all_data[item['audio_id']].update(item)

    # return a list of this
    return list(all_data.values())


def load_data(device, config):
    # 1. Load datasets + glove object
    load_path = config.load_path
    feature_set = config.feature_set

    task_num = 0

    total_data_size = 0

    # get a dict to hold all datasets
    all_datasets = {}

    # iterate through datasets listed in config file
    for dataset in config.datasets:
        dataset = dataset.lower()
        train_ds = pickle.load(open(f"{load_path}/{dataset}_{feature_set}_train.pickle", "rb"))
        dev_ds = pickle.load(open(f"{load_path}/{dataset}_{feature_set}_dev.pickle", "rb"))
        test_ds = pickle.load(open(f"{load_path}/{dataset}_{feature_set}_test.pickle", "rb"))
        clsswts = pickle.load(open(f"{load_path}/{dataset}_{feature_set}_clsswts.pickle", "rb"))

        dset_loss_func = torch.nn.CrossEntropyLoss(
            weight=clsswts.to(device) if config.model_params.use_clsswts else None,
            reduction="mean"
        )

        dset_obj = MultitaskObject(
            train_ds,
            dev_ds,
            test_ds,
            dset_loss_func,
            task_num=task_num
        )

        # add this to dict of datasets
        all_datasets[task_num] = dset_obj

        # increment task number
        task_num += 1

        # add to total data size
        total_data_size += len(train_ds)

    all_data_list = [all_datasets[item] for item in sorted(all_datasets.keys())]

    # optionally change loss multiplier
    if config.model_params.loss_multiplier:
        for obj in all_data_list:
            obj.change_loss_multiplier(len(obj.train_ds) / float(total_data_size))

    # if not using distilbert embeddings
    if not config.model_params.use_distilbert:
        # make glove
        glove_dict = make_glove_dict(config.glove_path)
        glove = Glove(glove_dict)

        # get set of pretrained embeddings and their shape
        pretrained_embeddings = glove.data
        num_embeddings = pretrained_embeddings.size()[0]
        print(f"shape of pretrained embeddings is: {glove.data.size()}")

    # create a single loss function
    if config.model_params.single_loss:
        loss_fx = torch.nn.CrossEntropyLoss(reduction="mean")
    else:
        loss_fx = None

    print(
        "Model, loss function, and optimization created"
    )

    # sampler = None
    # iif config.model_params.use_sampler:
    #    sampler = RandomOversampler(config.model_params.seed)
    # else:
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
    # allow to load a pretrained model for further fine-tuning
    if config.saved_model is not None:
        multitask_model.load_state_dict(torch.load(config.saved_model, map_location=device))
        multitask_model.to(device)

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

            train_multitask(data, loss_fx, sampler, device, output_path, config, num_embeddings, pretrained_embeddings)
