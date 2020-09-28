# implement training and testing for models
import sys
from collections import OrderedDict
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import parameters for model
from torch.utils.data import DataLoader

from models.bimodal_models import BimodalCNN
from models.parameters.earlyfusion_params import *
from models.plot_training import *

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


# adapted from https://github.com/joosthub/PyTorchNLPBook/blob/master/chapters/chapter_6/classifying-surnames/Chapter-6-Surname-Classification-with-RNNs.ipynb
def make_train_state(learning_rate, model_save_file):
    # makes a train state to save information on model during training/testing
    return {
        "stop_early": False,
        "early_stopping_step": 0,
        "early_stopping_best_val": 1e8,
        "learning_rate": learning_rate,
        "epoch_index": 0,
        "tasks": [],
        "train_loss": [],
        "train_acc": [],
        "train_avg_f1": {},
        "val_loss": [],
        "val_acc": [],
        "val_avg_f1": {},
        "best_val_loss": [],
        "best_val_acc": [],
        "best_loss": 100,
        "test_loss": -1,
        "test_acc": -1,
        "model_filename": model_save_file,
    }


def update_train_state(model, train_state):
    """Handle the training state updates.

    Components:
     - Early Stopping: Prevent overfitting.
     - Model Checkpoint: Model is saved if the model is better

    :param args: main arguments
    :param model: model to train
    :param train_state: a dictionary representing the training state values
    :returns:
        a new train_state
    """

    # Save one model at least
    if train_state["epoch_index"] == 0:
        torch.save(model.state_dict(), train_state["model_filename"])
        train_state["stop_early"] = False

        # use best validation accuracy for early stopping
        train_state["early_stopping_best_val"] = train_state["val_loss"][-1]
        # train_state['best_val_acc'] = train_state['val_acc'][-1]

    # Save model if performance improved
    elif train_state["epoch_index"] >= 1:
        loss_t = train_state["val_loss"][-1]

        # If loss worsened relative to BEST
        if loss_t >= train_state["early_stopping_best_val"]:
            # Update step
            train_state["early_stopping_step"] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state["early_stopping_best_val"]:
                torch.save(model.state_dict(), train_state["model_filename"])
                train_state["early_stopping_best_val"] = loss_t
                # train_state['best_val_acc'] = train_state['val_acc'][-1]

            # Reset early stopping step
            train_state["early_stopping_step"] = 0

        # Stop early ?
        train_state["stop_early"] = (
            train_state["early_stopping_step"] >= params.early_stopping_criteria
        )

    return train_state


def train_and_predict(
    classifier,
    train_state,
    train_ds,
    val_ds,
    batch_size,
    num_epochs,
    loss_func,
    optimizer,
    device="cpu",
    scheduler=None,
    sampler=None,
    avgd_acoustic=True,
    use_speaker=True,
    use_gender=False,
    binary=False,
    split_point=0.0,
):

    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # Iterate over training dataset
        running_loss = 0.0
        running_acc = 0.0

        # set classifier(s) to training mode
        classifier.train()

        batches = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, sampler=sampler
        )

        # set holders to use for error analysis
        ys_holder = []
        preds_holder = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # print(batch)
            # print(batch[0])
            # print(batch[1])
            # print(batch[2])
            # print(batch[3])
            # print(batch[4])
            # print(len(batch))
            # sys.exit()
            # get the gold labels
            y_gold = batch[4].to(device)
            # y_gold = batch[7].to(device)  # 4 is emotion, 5 is sentiment

            if split_point > 0:
                y_gold = torch.tensor(
                    [
                        1.0 if y_gold[i] > split_point else 0.0
                        for i in range(len(y_gold))
                    ]
                )
            # y_gold = batch.targets()

            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            batch_lengths = batch[-2].to(device)
            batch_acoustic_lengths = batch[-1].to(device)
            if use_speaker:
                batch_speakers = batch[2].to(device)
            else:
                batch_speakers = None

            if use_gender:
                batch_genders = batch[3].to(device)
            else:
                batch_genders = None

            if avgd_acoustic:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    gender_input=batch_genders,
                )
            else:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    acoustic_len_input=batch_acoustic_lengths,
                    gender_input=batch_genders,
                )

            if binary:
                y_pred = y_pred.float()
                y_gold = y_gold.float()

            # uncomment for prediction spot-checking during training
            # if epoch_index % 10 == 0:
            #     print(y_pred)
            #     print(y_gold)
            # if epoch_index == 35:
            #     sys.exit(1)
            # print("THE PREDICTIONS ARE: ")
            # print(y_pred)
            # print(y_gold)

            # add ys to holder for error analysis
            if binary:
                preds_holder.extend([round(item[0]) for item in y_pred.tolist()])
            else:
                preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
            ys_holder.extend(y_gold.tolist())

            # step 3. compute the loss
            # print(y_pred)
            # print(y_gold)
            # print(f"y-gold shape is: {y_gold.shape}")
            # print(y_pred)
            # print(f"y-pred shape is: {y_pred.shape}")
            loss = loss_func(y_pred, y_gold)
            loss_t = loss.item()  # loss for the item

            if len(list(y_pred.size())) > 1:
                if binary:
                    y_pred = torch.tensor([round(item[0]) for item in y_pred.tolist()])
                else:
                    # if type(y_gold[0]) == list or torch.is_tensor(y_gold[0]):
                    #     y_gold = torch.tensor([item.index(max(item)) for item in y_pred.tolist()])
                    y_pred = torch.tensor(
                        [item.index(max(item)) for item in y_pred.tolist()]
                    )
                    # print(y_gold)
                    # print(y_pred)
                    # print(type(y_gold))
                    # print(type(y_pred))
            else:
                y_pred = torch.round(y_pred)

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

            # compute the accuracy
            acc_t = torch.eq(y_pred, y_gold).sum().item() / len(y_gold)

            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # uncomment to see loss and accuracy measures for every minibatch
            # print("loss: {0}, running_loss: {1}, acc: {0}, running_acc: {1}".format(loss_t, running_loss,
            #                                                                       acc_t, running_acc))

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)
        train_state["train_acc"].append(running_acc)

        avg_f1 = precision_recall_fscore_support(
            ys_holder, preds_holder, average="weighted"
        )
        train_state["train_avg_f1"].append(avg_f1[2])
        # print("Training loss: {0}, training acc: {1}".format(running_loss, running_acc))
        print("Training weighted F-score: " + str(avg_f1))

        # Iterate over validation set--put it in a dataloader
        val_batches = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # reset loss and accuracy to zero
        running_loss = 0.0
        running_acc = 0.0

        # set classifier to evaluation mode
        classifier.eval()

        # set holders to use for error analysis
        ys_holder = []
        preds_holder = []

        # for each batch in the dataloader
        for batch_index, batch in enumerate(val_batches):
            # compute the output
            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            batch_lengths = batch[-2].to(device)
            batch_acoustic_lengths = batch[-1].to(device)
            if use_speaker:
                batch_speakers = batch[2].to(device)
            else:
                batch_speakers = None

            if use_gender:
                batch_genders = batch[3].to(device)
            else:
                batch_genders = None

            if avgd_acoustic:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    gender_input=batch_genders,
                )
            else:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    acoustic_len_input=batch_acoustic_lengths,
                    gender_input=batch_genders,
                )

            # get the gold labels
            # y_gold = batch[7].to(device)
            y_gold = batch[4].to(device)

            if split_point > 0:
                y_gold = torch.tensor(
                    [
                        1.0 if y_gold[i] > split_point else 0.0
                        for i in range(len(y_gold))
                    ]
                )
            # y_gold = batch.targets()

            if binary:
                y_pred = y_pred.float()
                y_gold = y_gold.float()

            # add ys to holder for error analysis
            if binary:
                preds_holder.extend([round(item[0]) for item in y_pred.tolist()])
            else:
                preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
            ys_holder.extend(y_gold.tolist())

            loss = loss_func(y_pred, y_gold)
            running_loss += (loss.item() - running_loss) / (batch_index + 1)

            # compute the loss
            if len(list(y_pred.size())) > 1:
                if binary:
                    y_pred = torch.tensor([round(item[0]) for item in y_pred.tolist()])
                else:
                    y_pred = torch.tensor(
                        [item.index(max(item)) for item in y_pred.tolist()]
                    )
            else:
                y_pred = torch.round(y_pred)

            # compute the accuracy
            acc_t = torch.eq(y_pred, y_gold).sum().item() / len(y_gold)
            running_acc += (acc_t - running_acc) / (batch_index + 1)

            # uncomment to see loss and accuracy for each minibatch
            # print("val_loss: {0}, running_val_loss: {1}, val_acc: {0}, running_val_acc: {1}".format(loss_t, running_loss,
            #                                                                       acc_t, running_acc))

        # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
        avg_f1 = precision_recall_fscore_support(
            ys_holder, preds_holder, average="weighted"
        )
        train_state["val_avg_f1"].append(avg_f1[2])
        print("Weighted F=score: " + str(avg_f1))

        # get confusion matrix
        if epoch_index % 5 == 0:
            print(confusion_matrix(ys_holder, preds_holder))
            print("Classification report: ")
            print(classification_report(ys_holder, preds_holder, digits=4))

        # add loss and accuracy to train state
        train_state["val_loss"].append(running_loss)
        train_state["val_acc"].append(running_acc)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break

#
# def single_dataset_multitask_train_and_predict(
#     classifier,
#     train_state,
#     train_ds_list,
#     val_ds_list,
#     batch_size,
#     num_epochs,
#     loss_func,
#     optimizer,
#     device="cpu",
#     scheduler=None,
#     sampler=None,
#     avgd_acoustic=True,
#     use_speaker=True,
#     use_gender=False,
# ):
#     """
#     Train_ds_list and val_ds_list are lists of datasets!
#     Each item in the list represents a dataset
#     Length of the list is the number of datasets used
#     todo: this is going to be tested with 2 tasks first, then more
#         it will be brittle at first, so make it more flexible
#     """
#     multi_dataset = (True if len(train_ds_list) > 1 else False)
#     # if len(train_ds_list) > 1:
#     #     multi_dataset = True
#     # else:
#     #     multi_dataset = False
#     num_tasks = len(train_ds_list)
#
#     for epoch_index in range(num_epochs):
#
#         print("Now starting epoch {0}".format(epoch_index))
#
#         train_state["epoch_index"] = epoch_index
#
#         # Iterate over training dataset
#         running_loss = 0.0
#
#         # set classifier(s) to training mode
#         classifier.train()
#
#         task_1_batches = DataLoader(
#             train_ds_list[0], batch_size=batch_size, shuffle=True, sampler=sampler
#         )
#         task_2_batches = DataLoader(
#             train_ds_list[1], batch_size=batch_size, shuffle=True, sampler=sampler
#         )
#
#         # set holders to use for error analysis
#         ys_holder = []
#         ys_2_holder = []
#         preds_holder = []
#         preds_2_holder = []
#
#         # for each batch in the list of batches created by the dataloader
#         for batch_index, batch in enumerate(task_1_batches):
#             # step 1. zero the gradients
#             optimizer.zero_grad()
#
#             # step 2. compute the output
#             y_pred, _, _, y_gold = get_batch_predictions(batch, classifier, 4, use_speaker, use_gender, avgd_acoustic, device)
#             _, y_2_pred, _, y_2_gold = get_batch_predictions(task_2_batches[batch_index], classifier, 4, use_speaker, use_gender, avgd_acoustic, device)
#
#             # add ys to holder for error analysis
#             preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
#             preds_2_holder.extend([item.index(max(item)) for item in y_2_pred.tolist()])
#             ys_holder.extend(y_gold.tolist())
#             ys_2_holder.extend(y_2_gold.tolist())
#
#             # step 3. compute the loss
#             class_1_loss = loss_func(y_pred, y_gold)
#             class_2_loss = loss_func(y_2_pred, y_2_gold)
#
#             loss = (class_1_loss / 1.6) + class_2_loss
#
#             loss_t = loss.item()  # loss for the item
#
#             # calculate running loss
#             running_loss += (loss_t - running_loss) / (batch_index + 1)
#
#             # step 4. use loss to produce gradients
#             loss.backward()
#
#             # step 5. use optimizer to take gradient step
#             optimizer.step()
#
#         # add loss and accuracy information to the train state
#         train_state["train_loss"].append(running_loss)
#
#         avg_f1 = precision_recall_fscore_support(
#             ys_holder, preds_holder, average="weighted"
#         )
#         train_state["train_avg_f1"].append(avg_f1[2])
#         # print("Training loss: {0}, training acc: {1}".format(running_loss, running_acc))
#         print("Training weighted F=score for EMOTION: " + str(avg_f1))
#
#         # Iterate over validation set--put it in a dataloader
#         val_1_batches = DataLoader(val_ds_list[0], batch_size=batch_size, shuffle=False)
#         val_2_batches = DataLoader(val_ds_list[1], batch_size=batch_size, shuffle=False)
#
#         # reset loss and accuracy to zero
#         running_loss = 0.0
#
#         # set classifier to evaluation mode
#         classifier.eval()
#
#         # set holders to use for error analysis
#         ys_holder = []
#         ys_2_holder = []
#         preds_holder = []
#         preds_2_holder = []
#
#         # for each batch in the dataloader
#         # todo: what if there are different numbers of batches? (diff dataset sizes)
#         for batch_index, batch in enumerate(val_1_batches):
#             # compute the output
#             y_pred, _, _, y_gold = get_batch_predictions(batch, classifier, 4, use_speaker, use_gender, avgd_acoustic, device)
#             _, y_2_pred, _, y_2_gold = get_batch_predictions(val_2_batches[batch_index], classifier, 4, use_speaker, use_gender, avgd_acoustic, device)
#
#             # add ys to holder for error analysis
#             preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
#             preds_2_holder.extend([item.index(max(item)) for item in y_2_pred.tolist()])
#             ys_holder.extend(y_gold.tolist())
#             ys_2_holder.extend(y_2_gold.tolist())
#
#             class_1_loss = loss_func(y_pred, y_gold)
#             class_2_loss = loss_func(y_2_pred, y_2_gold)
#
#             loss = (class_1_loss / 1.6) + class_2_loss
#
#             # loss = loss_func(ys_pred, ys_gold)
#             running_loss += (loss.item() - running_loss) / (batch_index + 1)
#
#             # uncomment to see loss and accuracy for each minibatch
#             # print("val_loss: {0}, running_val_loss: {1}, val_acc: {0}, running_val_acc: {1}".format(loss_t, running_loss,
#             #                                                                       acc_t, running_acc))
#
#         # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
#         avg_f1 = precision_recall_fscore_support(
#             ys_holder, preds_holder, average="weighted"
#         )
#         train_state["val_avg_f1"].append(avg_f1[2])
#         print("Weighted F=score: " + str(avg_f1))
#
#         # get confusion matrix
#         if epoch_index % 5 == 0:
#             print(confusion_matrix(ys_holder, preds_holder))
#             print("Classification report for EMOTION: ")
#             print(classification_report(ys_holder, preds_holder, digits=4))
#             print("======================================================")
#             print(confusion_matrix(ys_2_holder, preds_2_holder))
#             print("Classification report for SENTIMENT")
#             print(classification_report(ys_2_holder, preds_2_holder, digits=4))
#
#         # add loss and accuracy to train state
#         train_state["val_loss"].append(running_loss)
#
#         # update the train state now that our epoch is complete
#         train_state = update_train_state(model=classifier, train_state=train_state)
#
#         # update scheduler if there is one
#         if scheduler is not None:
#             scheduler.step(train_state["val_loss"][-1])
#
#         # if it's time to stop, end the training process
#         if train_state["stop_early"]:
#             break


def get_batch_predictions(batch, classifier, gold_idx, use_speaker=False,
                          use_gender=True, avgd_acoustic=True, device="cpu"):
    """
    Get the predictions for a batch
    batch: the batch of data from dataloader
    model: the model
    gold_idx: the index of gold labels used within the data batch
    returns predictions and gold labels for the batch
    """
    # get gold labels for the batch
    gold = batch[gold_idx].to(device)

    batch_acoustic = batch[0].to(device)
    batch_text = batch[1].to(device)
    batch_lengths = batch[6].to(device)
    # get acoustic lengths if necessary
    if not avgd_acoustic:
        batch_acoustic_lengths = batch[7].to(device)
    else:
        batch_acoustic_lengths = None
    # get speakers if necessary
    if use_speaker:
        batch_speakers = batch[2].to(device)
    else:
        batch_speakers = None
    # get gender if necessary
    if use_gender:
        batch_genders = batch[3].to(device)
    else:
        batch_genders = None

    y_pred, y_2_pred, y_3_pred = classifier(
        acoustic_input=batch_acoustic,
        text_input=batch_text,
        speaker_input=batch_speakers,
        length_input=batch_lengths,
        acoustic_len_input=batch_acoustic_lengths,
        gender_input=batch_genders,
    )

    return y_pred, y_2_pred, y_3_pred, gold




def test_model(
    classifier,
    test_ds,
    batch_size,
    loss_func,
    device="cpu",
    avgd_acoustic=True,
    use_speaker=True,
    use_gender=False,
):
    """
    Test a pretrained model
    """
    # Iterate over validation set--put it in a dataloader
    val_batches = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # reset loss and accuracy to zero
    running_loss = 0.0
    running_acc = 0.0

    # set classifier to evaluation mode
    classifier.eval()

    # set holders to use for error analysis
    ys_holder = []
    preds_holder = []

    # for each batch in the dataloader
    for batch_index, batch in enumerate(val_batches):
        # compute the output
        batch_acoustic = batch[0].to(device)
        batch_text = batch[1].to(device)
        batch_lengths = batch[-2].to(device)
        batch_acoustic_lengths = batch[-1].to(device)
        if use_speaker:
            batch_speakers = batch[2].to(device)
        else:
            batch_speakers = None

        if use_gender:
            batch_genders = batch[3].to(device)
        else:
            batch_genders = None

        if avgd_acoustic:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                gender_input=batch_genders,
            )
        else:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                acoustic_len_input=batch_acoustic_lengths,
                gender_input=batch_genders,
            )

        # get the gold labels
        y_gold = batch[4].to(device)

        # add ys to holder for error analysis
        preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
        ys_holder.extend(y_gold.tolist())

        y_pred = y_pred.float()
        y_gold = y_gold.float()

        loss = loss_func(y_pred, y_gold)
        running_loss += (loss.item() - running_loss) / (batch_index + 1)

        # compute the loss
        if len(list(y_pred.size())) > 1:
            y_pred = torch.tensor([item.index(max(item)) for item in y_pred.tolist()])
        else:
            y_pred = torch.round(y_pred)

        # compute the accuracy
        acc_t = torch.eq(y_pred, y_gold).sum().item() / len(y_gold)
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
    avg_f1 = precision_recall_fscore_support(
        ys_holder, preds_holder, average="weighted"
    )

    print("Weighted F=score: " + str(avg_f1))

    # get confusion matrix
    print(confusion_matrix(ys_holder, preds_holder))
    print("Classification report: ")
    print(classification_report(ys_holder, preds_holder, digits=4))


def multitask_train_and_predict(
    classifier,
    train_state,
    datasets_list,
    batch_size,
    num_epochs,
    optimizer,
    device="cpu",
    scheduler=None,
    sampler=None,
    avgd_acoustic=True,
    use_speaker=True,
    use_gender=False,
):
    """
    Train_ds_list and val_ds_list are lists of MultTaskObject objects!
    Length of the list is the number of datasets used
    """
    num_tasks = len(datasets_list)
    # get a list of the tasks by number
    for dset in datasets_list:
        train_state["tasks"].append(dset.task_num)
        train_state["train_avg_f1"][dset.task_num] = []
        train_state["val_avg_f1"][dset.task_num] = []

    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # Iterate over training dataset
        running_loss = 0.0

        # set classifier(s) to training mode
        classifier.train()

        batches, tasks = get_all_batches(datasets_list, batch_size=batch_size, shuffle=True)

        # set holders to use for error analysis
        ys_holder = {}
        for i in range(num_tasks):
            ys_holder[i] = []
        preds_holder = {}
        for i in range(num_tasks):
            preds_holder[i] = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # find the task for this batch
            batch_task = tasks[batch_index]

            # step 1. zero the gradients
            # zero all optimizers
            # for dataset in datasets_list:
            #     dataset.optimizer.zero_grad()

            optimizer.zero_grad()

            y_gold = batch[4].to(device)

            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            if use_speaker:
                batch_speakers = batch[2].to(device)
            else:
                batch_speakers = None

            if use_gender:
                batch_genders = batch[3].to(device)
            else:
                batch_genders = None
            batch_lengths = batch[-2].to(device)
            batch_acoustic_lengths = batch[-1].to(device)

            if avgd_acoustic:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    gender_input=batch_genders,
                    task_num=tasks[batch_index]
                )
            else:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    acoustic_len_input=batch_acoustic_lengths,
                    gender_input=batch_genders,
                    task_num=tasks[batch_index]
                )
            # print(y_pred)

            batch_pred = y_pred[batch_task]
            # print(f"y predictions are:\n{y_pred}")
            # print(f"y labels are:\n{y_gold}")

            if datasets_list[batch_task].binary:
                batch_pred = batch_pred.float()
                y_gold = y_gold.float()

            # calculate loss
            loss = datasets_list[batch_task].loss_fx(batch_pred, y_gold) * datasets_list[batch_task].loss_multiplier

            loss_t = loss.item()
            # print(f"Loss for this batch is: {loss_t}")

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # print(f"Running loss is now: {running_loss}")

            # use loss to produce gradients
            loss.backward()

            # add ys to holder for error analysis
            preds_holder[batch_task].extend([item.index(max(item)) for item in batch_pred.tolist()])
            ys_holder[batch_task].extend(y_gold.tolist())

            # increment optimizer
            optimizer.step()
            # for dataset in datasets_list:
            #     dataset.optimizer.step()

        # print(f"All predictions are:\n{preds_holder}")
        # print(f"All labels are:\n{ys_holder}")

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)

        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(ys_holder[task], preds_holder[task], average="weighted")
            print(f"Training weighted f-score for task {task}: {task_avg_f1}")
            train_state["train_avg_f1"][task].append(task_avg_f1[2])

        # Iterate over validation set--put it in a dataloader
        batches, tasks = get_all_batches(datasets_list, batch_size=batch_size, shuffle=True, partition="dev")

        # reset loss and accuracy to zero
        running_loss = 0.0

        # set classifier to evaluation mode
        classifier.eval()

        # set holders to use for error analysis
        ys_holder = {}
        for i in range(num_tasks):
            ys_holder[i] = []
        preds_holder = {}
        for i in range(num_tasks):
            preds_holder[i] = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # get the task for this batch
            batch_task = tasks[batch_index]

            y_gold = batch[4].to(device)

            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            if use_speaker:
                batch_speakers = batch[2].to(device)
            else:
                batch_speakers = None

            if use_gender:
                batch_genders = batch[3].to(device)
            else:
                batch_genders = None
            batch_lengths = batch[-2].to(device)
            batch_acoustic_lengths = batch[-1].to(device)

            # compute the output
            if avgd_acoustic:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    gender_input=batch_genders,
                    task_num=tasks[batch_index]
                )
            else:
                y_pred = classifier(
                    acoustic_input=batch_acoustic,
                    text_input=batch_text,
                    speaker_input=batch_speakers,
                    length_input=batch_lengths,
                    acoustic_len_input=batch_acoustic_lengths,
                    gender_input=batch_genders,
                    task_num=tasks[batch_index]
                )

            batch_pred = y_pred[batch_task]

            # print(f"Batch predictions are:\n{batch_pred}")
            # print(f"Batch labels are:\n{y_gold}")

            if datasets_list[batch_task].binary:
                batch_pred = batch_pred.float()
                y_gold = y_gold.float()

            # calculate loss
            loss = datasets_list[batch_task].loss_fx(batch_pred, y_gold) * datasets_list[batch_task].loss_multiplier

            loss_t = loss.item()
            # print(f"Loss for this batch is: {loss_t}")

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)
            # print(f"Running loss is: {running_loss}")

            # add ys to holder for error analysis
            preds_holder[batch_task].extend([item.index(max(item)) for item in batch_pred.tolist()])
            ys_holder[batch_task].extend(y_gold.tolist())

        # print(f"All evaluation predictions:\n{preds_holder}")
        # print(f"All evaluation labels:\n{ys_holder}")

        # print("Overall val loss: {0}, overall val acc: {1}".format(running_loss, running_acc))
        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(ys_holder[task], preds_holder[task], average="weighted")
            print(f"Val weighted f-score for task {task}: {task_avg_f1}")
            train_state["val_avg_f1"][task].append(task_avg_f1[2])

        if epoch_index % 5 == 0:
            for task in preds_holder.keys():
                print(f"Classification report and confusion matrix for task {task}:")
                print(confusion_matrix(ys_holder[task], preds_holder[task]))
                print("======================================================")
                print(classification_report(ys_holder[task], preds_holder[task], digits=4))

        # add loss and accuracy to train state
        train_state["val_loss"].append(running_loss)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break


def get_all_batches(dataset_list, batch_size, shuffle, partition="train"):
    """
    Create all batches and put them together as a single dataset
    """
    # set holder for batches
    all_batches = []
    all_loss_funcs = []

    # get number of tasks
    num_tasks = len(dataset_list)
    # print(f"The number of tasks is: {num_tasks}")

    # batch the data for each task
    for i in range(num_tasks):
        # print(f"Now printing partition {partition}")
        if partition == "train":
            data = DataLoader(dataset_list[i].train, batch_size=batch_size, shuffle=shuffle)
            # c = 0
            # for batch in data:
            #     c += 1
            #     print(batch)
            # print(f"Total number of batches in train partition: {c}")
        elif partition == "dev" or partition == "val":
            data = DataLoader(dataset_list[i].dev, batch_size=batch_size, shuffle=shuffle)
        elif partition == "test":
            data = DataLoader(dataset_list[i].test, batch_size=batch_size, shuffle=shuffle)
        else:
            sys.exit(f"Error: data partition {partition} not found")
        loss_func = dataset_list[i].loss_fx
        # print(f"The loss function is {loss_func}")
        all_batches.append(data)
        all_loss_funcs.append(loss_func)

    # print(f"Printing all batches prepared:")
    # i = 0
    # for batch in all_batches:
    #     i += 1
    #     print(batch)
    # print(f"Total number of batches: {i}")

    randomized_batches = []
    randomized_tasks = []

    task_num = 0
    for batches in all_batches:
        for i, batch in enumerate(batches):
            randomized_batches.append(batch)
            randomized_tasks.append(task_num)
        task_num += 1

    zipped = list(zip(randomized_batches, randomized_tasks))
    random.shuffle(zipped)
    randomized_batches, randomized_tasks = list(zip(*zipped))

    # print(f"Post-randomization all batches are:\n{randomized_batches}")
    # sys.exit()

    return randomized_batches, randomized_tasks


def predict_without_gold_labels(classifier,
    test_ds,
    batch_size,
    device="cpu",
    avgd_acoustic=True,
    use_speaker=True,
    use_gender=False,):
    """
    Test a pretrained model
    """
    # Iterate over validation set--put it in a dataloader
    test_batches = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # set classifier to evaluation mode
    classifier.eval()

    # set holders to use for error analysis
    preds_holder = []

    # for each batch in the dataloader
    for batch_index, batch in enumerate(test_batches):
        # compute the output
        batch_acoustic = batch[0].to(device)
        batch_text = batch[1].to(device)
        batch_lengths = batch[-2].to(device)
        batch_acoustic_lengths = batch[-1].to(device)
        if use_speaker:
            batch_speakers = batch[2].to(device)
        else:
            batch_speakers = None

        if use_gender:
            batch_genders = batch[3].to(device)
        else:
            batch_genders = None

        if avgd_acoustic:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                gender_input=batch_genders,
            )
        else:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                acoustic_len_input=batch_acoustic_lengths,
                gender_input=batch_genders,
            )

        # add ys to holder for error analysis
        preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])

    return preds_holder

