# implement training and testing for models
# this file contains the main training and evaluation functions
# used by our multimodal multitask networks
# Cheonkam has started to adapt a version of these functions
# for use with single-modal networks in train_and_test_single_models.py
import pickle
import sys
from datetime import datetime

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from tomcat_speech.training_and_evaluation_functions.train_and_test_utils import (
    get_all_batches,
    update_train_state,
    separate_data
)


def evaluate(
    classifier,
    train_state,
    datasets_list,
    batch_size,
    pickle_save_name,
    device="cpu",
    avgd_acoustic=True,
    use_speaker=True,
    use_gender=False,
    save_encoded_data=False
):
    """
    Train_ds_list and val_ds_list are lists of MultTaskObject objects!
    Length of the list is the number of datasets used
    :param classifier: a pytorch model for classification
    :param train_state: a train state made with function make_train_state
    :param datasets_list: a list of MultitaskObjects containing our data
    :param batch_size: the int size of minibatch
    :param pickle_save_name: the string name and path of a file
        where predictions may be saved
    :param device: 'cpu' or 'cuda'
    :param avgd_acoustic: whether averaged acoustic features are used
    :param use_speaker: whether speaker embeddings are included
    :param use_gender: whether speaker gender embeddings are included
    :param save_encoded_data: whether to save encoded predictions vectors
    """
    num_tasks = len(datasets_list)
    # get a list of the tasks by number
    for dset in datasets_list:
        train_state["tasks"].append(dset.task_num)
        train_state["test_avg_f1"][dset.task_num] = []

    # Iterate over validation set--put it in a dataloader
    batches, tasks = get_all_batches(
        datasets_list, batch_size=batch_size, shuffle=True, partition="test"
    )

    # set classifier to evaluation mode
    classifier.eval()

    # set holders to use for error analysis
    ys_holder = {}
    for i in range(num_tasks):
        ys_holder[i] = []
    preds_holder = {}
    for i in range(num_tasks):
        preds_holder[i] = []
    ids_holder = {}
    for i in range(num_tasks):
        ids_holder[i] = []
    preds_to_viz = {}
    for i in range(num_tasks):
        preds_to_viz[i] = []

    # for each batch in the list of batches created by the dataloader
    for batch_index, batch in enumerate(batches):
        # get the task for this batch
        batch_task = tasks[batch_index]

        batch_ids = batch['audio_id']
        ids_holder[batch_task].extend(batch_ids)

        batch_acoustic, batch_text, batch_speakers, batch_genders, y_gold, batch_lengths, batch_acoustic_lengths = separate_data(batch, device)

        if not use_speaker:
            batch_speakers = None
        if not use_gender:
            batch_genders = None

        # compute the output
        if avgd_acoustic:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                gender_input=batch_genders,
                task_num=tasks[batch_index],
                save_encoded_data=save_encoded_data
            )
        else:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                acoustic_len_input=batch_acoustic_lengths,
                gender_input=batch_genders,
                task_num=tasks[batch_index],
                save_encoded_data=save_encoded_data
            )

        if save_encoded_data:
            encoded_data = y_pred[-1]

        batch_pred = y_pred[batch_task]

        if datasets_list[batch_task].binary:
            batch_pred = batch_pred.float()
            y_gold = y_gold.float()

        # add preds vector to holder for pca visualization
        if save_encoded_data:
            preds_to_viz[batch_task].extend(encoded_data.tolist())

        # add ys to holder for error analysis
        preds_holder[batch_task].extend(
            [item.index(max(item)) for item in batch_pred.tolist()]
        )
        ys_holder[batch_task].extend(y_gold.tolist())

    for task in preds_holder.keys():
        task_avg_f1 = precision_recall_fscore_support(
            ys_holder[task], preds_holder[task], average="weighted"
        )
        print(f"Test weighted f-score for task {task}: {task_avg_f1}")
        train_state["test_avg_f1"][task].append(task_avg_f1[2])

    for task in preds_holder.keys():
        print(f"Classification report and confusion matrix for task {task}:")
        print(confusion_matrix(ys_holder[task], preds_holder[task]))
        print("======================================================")
        print(classification_report(ys_holder[task], preds_holder[task], digits=4))

    # combine
    gold_preds_ids = {}
    for task in preds_holder.keys():
        gold_preds_ids[task] = list(
            zip(ys_holder[task], preds_holder[task], ids_holder[task])
        )

    # save to pickle
    with open(pickle_save_name, "wb") as pfile:
        pickle.dump(gold_preds_ids, pfile)

    if save_encoded_data:
        with open("output/encoded_output.pickle", 'wb') as pf:
            pickle.dump(preds_to_viz, pf)


def train_and_predict(
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
    save_encoded_data=False,
    loss_fx=None,
    use_spec=False
):
    """
    Perform training and prediction on train and dev set
    This function is called from model training scripts
    :param classifier: A pytorch model
    :param train_state: a train state created by make_train_state
    :param datasets_list: a list of MultitaskObjects containing our data
    :param batch_size: the int size of minibatch
    :param num_epochs: the max number of epochs for training
    :param optimizer: an optimizer for training
    :param device: 'cpu' or 'cuda'
    :param scheduler: a scheduler; this isn't currently in use
    :param sampler: None or a string
        currently our model only uses 'oversampling'
    :param avgd_acoustic: whether averaged acoustic features are used
    :param use_speaker: whether speaker embeddings are included
    :param use_gender: whether speaker gender embeddings are included
    :param save_encoded_data: whether to save encoded predictions vectors
    :param loss_fx: a loss function; with default None, the loss
        function saved within each MultitaskObject from param
        datasets_list is used rather than this
    :param use_spec: whether to include spectrograms
    """
    num_tasks = len(datasets_list)

    print(f"Number of tasks: {num_tasks}")
    # get a list of the tasks by number
    for dset in datasets_list:
        train_state["tasks"].append(dset.task_num)
        train_state["train_avg_f1"][dset.task_num] = []
        train_state["val_avg_f1"][dset.task_num] = []
        train_state["val_best_f1"].append(0)

    for epoch_index in range(num_epochs):

        first = datetime.now()
        print(f"Starting epoch {epoch_index} at {first}")

        train_state["epoch_index"] = epoch_index

        # get running loss, holders of ys and predictions on training partition
        running_loss, ys_holder, preds_holder = run_model(
            datasets_list,
            classifier,
            batch_size,
            num_tasks,
            device,
            use_speaker,
            use_gender,
            avgd_acoustic,
            optimizer,
            mode="training",
            loss_fx=loss_fx,
            sampler=sampler,
            use_spec=use_spec,
        )

        # add loss and accuracy to train state
        train_state["train_loss"].append(running_loss)

        # get precision, recall, f1 info
        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="weighted"
            )
            print(f"Training weighted f-score for task {task}: {task_avg_f1}")
            # add training f1 to train state
            train_state["train_avg_f1"][task].append(task_avg_f1[2])

        # get running loss, holders of ys and predictions on dev partition
        running_loss, ys_holder, preds_holder = run_model(
            datasets_list,
            classifier,
            batch_size,
            num_tasks,
            device,
            use_speaker,
            use_gender,
            avgd_acoustic,
            optimizer,
            mode="eval",
            loss_fx=loss_fx,
            use_spec=use_spec
        )

        # get precision, recall, f1 info
        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="weighted"
            )
            print(f"Val weighted f-score for task {task}: {task_avg_f1}")
            # add val f1 to train state
            train_state["val_avg_f1"][task].append(task_avg_f1[2])
            if task_avg_f1[2] > train_state["val_best_f1"][task]:
                train_state["val_best_f1"][task] = task_avg_f1[2]

        # every 5 epochs, print out a classification report on validation set
        if epoch_index % 5 == 0:
            for task in preds_holder.keys():
                print(f"Classification report and confusion matrix for task {task}:")
                print(confusion_matrix(ys_holder[task], preds_holder[task]))
                print("======================================================")
                print(
                    classification_report(ys_holder[task], preds_holder[task], digits=4)
                )

        # add loss and accuracy to train state
        train_state["val_loss"].append(running_loss)

        # update the train state now that our epoch is complete
        train_state = update_train_state(model=classifier, train_state=train_state, optimizer=optimizer)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break

        # print out how long this epoch took
        last = datetime.now()
        print(f"This epoch took {last - first}")
        sys.stdout.flush()


def run_model(
    datasets_list,
    classifier,
    batch_size,
    num_tasks,
    device,
    use_speaker,
    use_gender,
    avgd_acoustic,
    optimizer,
    mode="training",
    loss_fx=None,
    sampler=None,
    use_spec=False
):
    """
    Run the model in either training or testing within a single epoch
    This model is called from within function train_and_predict
    :param datasets_list: a list of MultitaskObjects containing our data
    :param classifier: a pytorch classifier
    :param batch_size: number of items in each minibatch
    :param num_tasks: the number of tasks we are getting predictions for
    :param device: 'cpu' or 'cuda'
    :param use_speaker: whether to include speaker embeddings
    :param use_gender: whether to include speaker gender embeddings
    :param avgd_acoustic: whether acoustic features are averaged
    :param optimizer: an optimizer
    :param mode: 'training' or 'evaluation'
    :param loss_fx: a loss function; with default None, the loss
        function saved within each MultitaskObject from param
        datasets_list is used rather than this
    :param sampler: None or a string
        currently our model only uses 'oversampling'
    :param use_spec: whether to include spectrograms
    :return: running_loss, gold labels, and predictions
    """
    # Iterate over training dataset
    running_loss = 0.0

    # set classifier(s) to appropriate mode
    if mode.lower() == "training" or mode.lower() == "train":
        classifier.train()
        batches, tasks = get_all_batches(
            datasets_list, batch_size=batch_size, shuffle=True, sampler=sampler
        )
    else:
        classifier.eval()
        batches, tasks = get_all_batches(
            datasets_list, batch_size=batch_size, shuffle=True, partition="dev"
        )

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

        # zero gradients
        if mode.lower() == "training" or mode.lower() == "train":
            optimizer.zero_grad()

        # get ys and predictions for the batch
        y_gold, batch_pred = get_predictions(
            batch,
            batch_task,
            classifier,
            device,
            use_speaker,
            use_gender,
            avgd_acoustic,
            use_spec=use_spec,
        )

        # calculate loss
        if loss_fx:
            loss = loss_fx(batch_pred, y_gold) * datasets_list[batch_task].loss_multiplier
        else:
            loss = (
                datasets_list[batch_task].loss_fx(batch_pred, y_gold)
                * datasets_list[batch_task].loss_multiplier
            )
        loss_t = loss.item()

        # calculate running loss
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # use loss to produce gradients
        if mode.lower() == "training" or mode.lower() == "train":
            loss.backward()

        # add ys to holder for error analysis
        preds_holder[batch_task].extend(
            [item.index(max(item)) for item in batch_pred.detach().tolist()]
        )
        ys_holder[batch_task].extend(y_gold.detach().tolist())

        # increment optimizer
        if mode.lower() == "training" or mode.lower() == "train":
            optimizer.step()

    return running_loss, ys_holder, preds_holder


def get_predictions(
    batch,
    batch_task,
    classifier,
    device,
    use_speaker,
    use_gender,
    avgd_acoustic,
    use_spec=False,
):
    """
    Get predictions from data
    This function is called from within run_model
    :param batch: a batch of data
    :param batch_task: the task number for the current batch
    :param classifier: a classifier for making predictions
    :param device: 'cpu' or 'cuda'
    :param use_speaker: whether to include speaker embeddings
    :param use_gender: whether to include speaker gender embeddings
    :param avgd_acoustic: whether acoustic features are averaged
    :param use_spec: whether to include spectrograms
    :return: gold labels, predictions for the batch
    """
    # get parts of batches
    # get data
    if batch_task is not None:
        y_gold = batch["ys"][0].detach().to(device)
    else:
        y_gold = batch["ys"].detach().to(device)

    batch_acoustic = batch["x_acoustic"].detach().to(device)
    batch_text = batch["x_utt"].detach().to(device)
    if use_speaker:
        batch_speakers = batch["x_speaker"].to(device)
    else:
        batch_speakers = None

    if use_gender:
        batch_genders = batch["x_gender"].to(device)
    else:
        batch_genders = None
    batch_lengths = batch["utt_length"]
    batch_acoustic_lengths = batch["acoustic_length"]
    if use_spec:
        batch_spec = batch["x_spec"].to(device)
    else:
        batch_spec = None

    # feed these parts into classifier
    # compute the output
    if avgd_acoustic:
        y_pred = classifier(
            acoustic_input=batch_acoustic,
            text_input=batch_text,
            spec_input=batch_spec,
            speaker_input=batch_speakers,
            length_input=batch_lengths,
            gender_input=batch_genders,
            task_num=batch_task,
        )
    else:
        y_pred = classifier(
            acoustic_input=batch_acoustic,
            text_input=batch_text,
            spec_input=batch_spec,
            speaker_input=batch_speakers,
            length_input=batch_lengths,
            acoustic_len_input=batch_acoustic_lengths,
            gender_input=batch_genders,
            task_num=batch_task,
        )

    if batch_task is not None:
        batch_pred = y_pred[batch_task]
    else:
        batch_pred = y_pred

    return y_gold, batch_pred
