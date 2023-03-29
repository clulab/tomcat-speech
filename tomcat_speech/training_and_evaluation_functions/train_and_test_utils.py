
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import sys

from tomcat_speech.models.multimodal_models import MultitaskModel, MultitaskAcousticShared, \
    MultitaskDuplicateInputModel, MultitaskTextShared


def set_cuda_and_seeds(config):
    """
    Set cuda and random seeds
    :param config: a config file containing random seed number
    """
    # set cuda
    cuda = False
    if torch.cuda.is_available():
        cuda = True

    device = torch.device("cuda" if cuda else "cpu")

    # # Check CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(2)

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

    return device


def select_model(model_params, num_embeddings, pretrained_embeddings, multidataset=True):
    """
    Use model parameters to select the appropriate model
    Return this model for training
    :param model_params: model parameters from a config file
    :param num_embeddings: the number of embeddings (if necessary)
    :param pretrained_embeddings: pretrained GloVe embeddings, if using
    :param multidataset: whether data is coming from multiple datasets
    :return: a pytorch model
    """
    # set embeddings to None if using bert -- they are calculated
    #   anyway, so if you don't do this, it will ALWAYS use embeddings
    if model_params.use_distilbert:
        num_embeddings = None
        pretrained_embeddings = None

    if "acoustic_shared" in model_params.model.lower():
        model = MultitaskAcousticShared(params=model_params,
                                        use_distilbert=model_params.use_distilbert,
                                        num_embeddings=num_embeddings,
                                        pretrained_embeddings=pretrained_embeddings)
    elif "text_shared" in  model_params.model.lower():
        model = MultitaskTextShared(params=model_params,
                                    use_distilbert=model_params.use_distilbert,
                                    num_embeddings=num_embeddings,
                                    pretrained_embeddings=pretrained_embeddings)
    elif "duplicate_input" in model_params.model.lower():
        model = MultitaskDuplicateInputModel(params=model_params,
                                             use_distilbert=model_params.use_distilbert,
                                             num_embeddings=num_embeddings,
                                             pretrained_embeddings=pretrained_embeddings)
    else:
        model = MultitaskModel(params=model_params,
                               use_distilbert=model_params.use_distilbert,
                               num_embeddings=num_embeddings,
                               pretrained_embeddings=pretrained_embeddings,
                               multi_dataset=multidataset)

    return model


def get_all_batches(dataset_list, batch_size, shuffle, partition="train", sampler=None):
    """
    Create all batches and collate them
    :param dataset_list: a list of MultitaskObjects containing the data
    :param batch_size: the size of each minibatch
    :param shuffle: bool for whether to shuffle the order of data points
    :param sampler: None or a string ('oversampling' currently)
    """
    # set holder for batches
    all_batches = []
    all_loss_funcs = []

    # get number of tasks
    num_tasks = len(dataset_list)

    # batch the data for each task
    for i in range(num_tasks):
        if partition == "train":
            # (over)sample training data if needed
            if sampler is not None:
                dataset_list[i].train = sampler.prep_data_through_oversampling(dataset_list[i].train)
            data = DataLoader(
                dataset_list[i].train, batch_size=batch_size, shuffle=shuffle
            )
        elif partition == "dev" or partition == "val":
            data = DataLoader(
                dataset_list[i].dev, batch_size=batch_size, shuffle=shuffle
            )
        elif partition == "test":
            data = DataLoader(
                dataset_list[i].test, batch_size=batch_size, shuffle=shuffle
            )
        else:
            sys.exit(f"Error: data partition {partition} not found")
        loss_func = dataset_list[i].loss_fx
        # put batches together
        all_batches.append(data)
        all_loss_funcs.append(loss_func)

    randomized_batches = []
    randomized_tasks = []

    # randomize batches
    task_num = 0
    for batches in all_batches:
        for i, batch in enumerate(batches):
            randomized_batches.append(batch)
            randomized_tasks.append(task_num)
        task_num += 1

    zipped = list(zip(randomized_batches, randomized_tasks))
    random.shuffle(zipped)
    randomized_batches, randomized_tasks = list(zip(*zipped))

    return randomized_batches, randomized_tasks


def make_train_state(learning_rate, model_save_file, early_stopping_criterion):
    """
    Make a train state to save information on model during training/testing
    :param learning_rate: the learning rate for the model
    :param model_save_file: the string name of a file to save the model
        during training
    :param early_stopping_criterion: the patience of the model
        used to determine when to stop early
    """
    return {
        "stop_early": False,
        "early_stopping_step": 0,
        "early_stopping_best_val": 0.0,
        "learning_rate": learning_rate,
        "epoch_index": 0,
        "tasks": [],
        "train_loss": [],
        "train_acc": [],
        "train_avg_f1": {},
        "val_loss": [],
        "val_acc": [],
        "val_avg_f1": {},
        "val_best_f1": [],
        "best_val_loss": [],
        "best_val_acc": [],
        "test_avg_f1": {},
        "best_loss": 100,
        "test_loss": -1,
        "test_acc": -1,
        "model_filename": model_save_file,
        "early_stopping_criterion": early_stopping_criterion,
    }


def update_train_state(model, train_state, optimizer=None):
    """Handle the training state updates.
    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better
    :param model: the model used in training
    :param train_state: a dictionary representing the training state values
    :param optimizer: the optimizer used in training
        only passed as a parameter if we want to save the
        optimizer's state dict alongside the model
    :returns:
        an updated train_state
    """
    # Save one model at least
    if train_state["epoch_index"] == 0:
        if optimizer is not None:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        train_state["model_filename"])
        else:
            torch.save(model.state_dict(), train_state["model_filename"])
        train_state["stop_early"] = False

        # use val f1 instead of val_loss
        avg_f1_t = 0
        for item in train_state["val_avg_f1"].values():
            avg_f1_t += item[-1]
        avg_f1_t = avg_f1_t / len(train_state["tasks"])

        # use best validation accuracy for early stopping
        train_state["early_stopping_best_val"] = avg_f1_t

    # Save model if performance improved
    elif train_state["epoch_index"] >= 1:
        # use val f1 instead of val_loss
        avg_f1_t = 0
        for item in train_state["val_avg_f1"].values():
            avg_f1_t += item[-1]
        avg_f1_t = avg_f1_t / len(train_state["tasks"])

        # if avg f1 is higher
        if avg_f1_t >= train_state["early_stopping_best_val"]:
            # save this as best model
            if optimizer is not None:
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            train_state["model_filename"])
            else:
                torch.save(model.state_dict(), train_state["model_filename"])
            print("updating model")
            train_state["early_stopping_best_val"] = avg_f1_t
            train_state["early_stopping_step"] = 0
        else:
            train_state["early_stopping_step"] += 1

        # Stop early ?
        train_state["stop_early"] = (
            train_state["early_stopping_step"] >= train_state["early_stopping_criterion"]
        )

    return train_state


def separate_data(batch_of_data, device):
    """
    Separate data into the useable parts
    Accepts data as a list of tensors/lists or as a dict
    If data is a list:
    data[0] == acoustic features
    data[1] == text features
    data[2] == speakers
    data[3] == speaker genders
    data[4] == gold labels
    data[-1] == acoustic lengths
    data[-2] == text lengths
    Depending on dataset, total length of list may vary
    """
    if type(batch_of_data) is not dict:
        batch_acoustic = batch_of_data[0].detach().to(device)
        batch_text = batch_of_data[1].detach().to(device)
        batch_speakers = batch_of_data[2].to(device)
        batch_genders = batch_of_data[3].to(device)
        y_gold = batch_of_data[4].detach().to(device)
        batch_lengths = batch_of_data[-2].to(device)
        batch_acoustic_lengths = batch_of_data[-1].to(device)
    else:
        batch_acoustic = batch_of_data["x_acoustic"].detach().to(device)
        batch_text = batch_of_data["x_utt"].detach().to(device)
        batch_speakers = batch_of_data["x_speaker"].to(device)
        batch_genders = batch_of_data["x_gender"].to(device)
        # todo add flexibilty for other tasks in same dataset
        y_gold = batch_of_data["ys"][0].detach().to(device)
        batch_lengths = batch_of_data["utt_length"].to(device)
        batch_acoustic_lengths = batch_of_data["acoustic_length"].to(device)

    return (
        batch_acoustic,
        batch_text,
        batch_speakers,
        batch_genders,
        y_gold,
        batch_lengths,
        batch_acoustic_lengths,
    )
