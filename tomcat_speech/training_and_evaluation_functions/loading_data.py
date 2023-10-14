# contains functions to load preprocessed data
from tomcat_speech.data_prep.samplers import RandomOversampler
from tomcat_speech.data_prep.utils.data_prep_helpers import (
    MultitaskObject,
    Glove,
    make_glove_dict,
)

import pickle
import torch
import pandas as pd
import sqlite3


def load_alter_modality_data(device, config, use_text=True, use_acoustic=True):
    """
    Load the modality-separated data
    :param device: 'cpu' or 'cuda'
    :param config: the config file containing parameters
    :param use_text: whether to include text modality
    :param use_acoustic: whether to include acoustic modality
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
            spec_base = (
                f"{load_path}/field_separated_data/spectrogram_data/{dataset}_spec"
            )
            print("Loading spec features for train set")
            train_spec = pickle.load(open(f"{spec_base}_train.pickle", "rb"))
            print("Spec features loaded for train set")

            longest = 300
            x_replacer = torch.zeros(longest, 513)
            for item in train_spec:
                x = x_replacer.detach().clone()
                x[: min(len(item["x_spec"]), longest), :513] = item["x_spec"][
                    : min(len(item["x_spec"]), longest)
                ]
                item["x_spec"] = x
            train_modalities.append(train_spec)
            del train_spec

            print("Loading spec features for dev set")
            dev_spec = pickle.load(open(f"{spec_base}_dev.pickle", "rb"))
            for item in dev_spec:
                x = x_replacer.detach().clone()
                x[: min(len(item["x_spec"]), longest), :513] = item["x_spec"][
                    : min(len(item["x_spec"]), longest)
                ]
                item["x_spec"] = x
            dev_modalities.append(dev_spec)
            del dev_spec

            print("loading spec features for test set")
            test_spec = pickle.load(open(f"{spec_base}_test.pickle", "rb"))
            for item in test_spec:
                x = x_replacer.detach().clone()
                x[: min(len(item["x_spec"]), longest), :513] = item["x_spec"][
                    : min(len(item["x_spec"]), longest)
                ]
                item["x_spec"] = x
            test_modalities.append(test_spec)
            del test_spec

        # add ys data and classweights to this
        print(f"Loading gold labels for dataset {dataset}")
        ys_base = f"{load_path}/field_separated_data/ys_data/{dataset}_ys"
        ys_train = pickle.load(open(f"{ys_base}_train.pickle", "rb"))
        ys_dev = pickle.load(open(f"{ys_base}_dev.pickle", "rb"))
        ys_test = pickle.load(open(f"{ys_base}_test.pickle", "rb"))

        train_modalities.append(ys_train)
        dev_modalities.append(ys_dev)
        test_modalities.append(ys_test)

        del ys_train, ys_dev, ys_test

        # combine modalities
        if len(train_modalities) < 2:
            exit(
                "No modalities data has been loaded; please select at least one modality to load"
            )
        if dataset == "asist":
            train_data = combine_modality_data(train_modalities, preserve_uuids=True)
            dev_data = combine_modality_data(dev_modalities, preserve_uuids=True)
            test_data = combine_modality_data(test_modalities, preserve_uuids=True)

            train_uuids, dev_uuids, test_uuids = get_multicat_train_test_bytrial()

            train_data, dev_data, test_data = reorganize_data(train_data, dev_data,
                                                              test_data, train_uuids,
                                                              dev_uuids, test_uuids)
        else:
            train_data = combine_modality_data(train_modalities)
            dev_data = combine_modality_data(dev_modalities)
            test_data = combine_modality_data(test_modalities)

        del train_modalities, dev_modalities, test_modalities

        clsswts_base = (
            f"{load_path}/field_separated_data/class_weights/{dataset}_clsswts.pickle"
        )
        clsswts = pickle.load(open(clsswts_base, "rb"))

        dset_loss_func = torch.nn.CrossEntropyLoss(
            weight=clsswts.to(device) if config.model_params.use_clsswts else None,
            reduction="mean",
        )

        all_datasets[task_num] = MultitaskObject(
            train_data, dev_data, test_data, dset_loss_func, task_num=task_num
        )
        # increment task number
        task_num += 1

        # add to the total data size
        total_data_size += len(train_data)

    all_data_list = [all_datasets[item] for item in sorted(all_datasets.keys())]

    del all_datasets

    # optionally change loss multiplier
    if config.model_params.loss_multiplier:
        for obj in all_data_list:
            obj.change_loss_multiplier(len(obj.train) / float(total_data_size))

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

    print("Model, loss function, and optimization created")

    # set sampler
    if config.model_params.use_sampler:
        sampler = RandomOversampler(config.model_params.seed)
    else:
        sampler = None

    if not config.model_params.use_distilbert:
        return all_data_list, loss_fx, sampler, num_embeddings, pretrained_embeddings
    else:
        return all_data_list, loss_fx, sampler


def load_modality_data(device, config, use_text=True, use_acoustic=True):
    """
    Load the modality-separated data
    :param device: 'cpu' or 'cuda'
    :param config: the config file containing parameters
    :param use_text: whether to include text modality
    :param use_acoustic: whether to include acoustic modality
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
            spec_base = (
                f"{load_path}/field_separated_data/spectrogram_data/{dataset}_spec"
            )
            print("Loading spec features for train set")
            train_spec = pickle.load(open(f"{spec_base}_train.pickle", "rb"))
            print("Spec features loaded for train set")

            longest = 300
            x_replacer = torch.zeros(longest, 513)
            for item in train_spec:
                x = x_replacer.detach().clone()
                x[: min(len(item["x_spec"]), longest), :513] = item["x_spec"][
                    : min(len(item["x_spec"]), longest)
                ]
                item["x_spec"] = x
            train_modalities.append(train_spec)
            del train_spec

            print("Loading spec features for dev set")
            dev_spec = pickle.load(open(f"{spec_base}_dev.pickle", "rb"))
            for item in dev_spec:
                x = x_replacer.detach().clone()
                x[: min(len(item["x_spec"]), longest), :513] = item["x_spec"][
                    : min(len(item["x_spec"]), longest)
                ]
                item["x_spec"] = x
            dev_modalities.append(dev_spec)
            del dev_spec

            print("loading spec features for test set")
            test_spec = pickle.load(open(f"{spec_base}_test.pickle", "rb"))
            for item in test_spec:
                x = x_replacer.detach().clone()
                x[: min(len(item["x_spec"]), longest), :513] = item["x_spec"][
                    : min(len(item["x_spec"]), longest)
                ]
                item["x_spec"] = x
            test_modalities.append(test_spec)
            del test_spec

        # add ys data and classweights to this
        print(f"Loading gold labels for dataset {dataset}")
        ys_base = f"{load_path}/field_separated_data/ys_data/{dataset}_ys"
        ys_train = pickle.load(open(f"{ys_base}_train.pickle", "rb"))
        ys_dev = pickle.load(open(f"{ys_base}_dev.pickle", "rb"))
        ys_test = pickle.load(open(f"{ys_base}_test.pickle", "rb"))

        train_modalities.append(ys_train)
        dev_modalities.append(ys_dev)
        test_modalities.append(ys_test)

        del ys_train, ys_dev, ys_test

        # combine modalities
        if len(train_modalities) < 2:
            exit(
                "No modalities data has been loaded; please select at least one modality to load"
            )

        train_data = combine_modality_data(train_modalities)
        dev_data = combine_modality_data(dev_modalities)
        test_data = combine_modality_data(test_modalities)

        del train_modalities, dev_modalities, test_modalities

        clsswts_base = (
            f"{load_path}/field_separated_data/class_weights/{dataset}_clsswts.pickle"
        )
        clsswts = pickle.load(open(clsswts_base, "rb"))

        dset_loss_func = torch.nn.CrossEntropyLoss(
            weight=clsswts.to(device) if config.model_params.use_clsswts else None,
            reduction="mean",
        )

        all_datasets[task_num] = MultitaskObject(
            train_data, dev_data, test_data, dset_loss_func, task_num=task_num
        )
        # increment task number
        task_num += 1

        # add to the total data size
        total_data_size += len(train_data)

    all_data_list = [all_datasets[item] for item in sorted(all_datasets.keys())]

    del all_datasets

    # optionally change loss multiplier
    if config.model_params.loss_multiplier:
        for obj in all_data_list:
            obj.change_loss_multiplier(len(obj.train) / float(total_data_size))

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

    print("Model, loss function, and optimization created")

    # set sampler
    if config.model_params.use_sampler:
        sampler = RandomOversampler(config.model_params.seed)
    else:
        sampler = None

    if not config.model_params.use_distilbert:
        return all_data_list, loss_fx, sampler, num_embeddings, pretrained_embeddings
    else:
        return all_data_list, loss_fx, sampler


def get_multicat_train_test_bytrial():
    # get list of uuids for train, dev, test set with following partitions
    train_trials = ["T000603", "T000604", "T000611", "T000612", "T000620",
                    "T000622", "T000623", "T000624", "T000627", "T000628",
                    "T000631", "T000632", "T000635", "T000636", "T000637",
                    "T000638", "T000703", "T000704", "T000713", "T000714",
                    "T000715", "T000716", "T000719", "T000720", "T000723",
                    "T000724", "T000729", "T000730"]
    dev_trials = ["T000607", "T000608", "T000613", "T000633", "T000634"]
    test_trials = ["T000605", "T000606", "T000609", "T000610", "T000625",
                   "T000626", "T000627", "T000671", "T000672", "T000727",
                   "T000728", "T000737", "T000738"]

    conn = sqlite3.connect("../../../sqlite_db_creation/multicat.db")
    multicat = pd.read_sql("SELECT * FROM utterance WHERE emotion IS NOT NULL", conn)

    multicat_train = multicat[multicat['trial'].isin(train_trials)]['original_uuid'].tolist()
    multicat_dev = multicat[multicat['trial'].isin(dev_trials)]['original_uuid'].tolist()
    multicat_test = multicat[multicat['trial'].isin(test_trials)]['original_uuid'].tolist()

    return multicat_train, multicat_dev, multicat_test


def combine_modality_data(list_of_modality_data, preserve_uuids=False):
    """
    Use a list of lists of dicts (each of which contains info on a modality)
    to get a single list of dicts for the dataset
    return this single list of dicts
    :param preserve_uuids: whether to keep the uuids (used if further data
        alterations are going to take place)
    """
    all_data = {}

    # get all utterance IDs
    for item in list_of_modality_data[0]:
        all_data[item["audio_id"]] = item

    for dataset in list_of_modality_data[1:]:
        for item in dataset:
            all_data[item["audio_id"]].update(item)

    if not preserve_uuids:
        # return a list of this
        return list(all_data.values())
    else:
        return all_data


def reorganize_data(train_data, dev_data, test_data, train_uuids, dev_uuids, test_uuids):
    """
    Reorganize data partitions so that they use the specific partitions necessary
    """
    train = []
    dev = []
    test = []

    for item in train_data.keys():
        if item in train_uuids:
            train.append(train_data['item'])
        elif item in dev_uuids:
            dev.append(train_data['item'])
        elif item in test_uuids:
            test.append(train_data['item'])

    for item in dev_data.keys():
        if item in train_uuids:
            train.append(dev_data['item'])
        elif item in dev_uuids:
            dev.append(dev_data['item'])
        elif item in test_uuids:
            test.append(dev_data['item'])

    for item in test_data.keys():
        if item in train_uuids:
            train.append(test_data['item'])
        elif item in dev_uuids:
            dev.append(test_data['item'])
        elif item in test_uuids:
            test.append(test_data['item'])

    return train, dev, test
