# implement training and testing for models
# this code is being used with train_single_task and train_single_task_grid_search
# implementation incomplete as of 2023.03.28
import sys
import random
from datetime import datetime

# import parameters for model
from torch.utils.data import DataLoader, RandomSampler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from tomcat_speech.models.train_and_test_models import update_train_state


def get_data_from_loader(
    dataset_list, batch_size, shuffle, device, partition="train", sampler=None
):
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
                dataset_list[i].train = sampler.prep_data_through_oversampling(
                    dataset_list[i].train
                )
            data = DataLoader(
                dataset_list[i].train,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                # pin_memory=True with GPU, not CPU
                # added 12/19/22 for testing parallel processing
            )
        elif partition == "dev" or partition == "val":
            data = DataLoader(
                dataset_list[i].dev,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                # added 12/19/22
            )
        elif partition == "test":
            data = DataLoader(
                dataset_list[i].test,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                # added 12/19/22
            )
        else:
            sys.exit(f"Error: data partition {partition} not found")
        loss_func = dataset_list[i].loss_fx
        # put batches together
        all_batches.append(data)
        all_loss_funcs.append(loss_func)

    return all_batches, all_loss_funcs


def randomize_batches(all_batches):
    """
    Randomize the order in which minibatches are seen
    Done to get different batches of same task separated for multitasking
    """
    randomized_batches = []
    randomized_tasks = []

    # randomize batches
    task_num = 0
    for batches in all_batches:
        for i, batch in enumerate(batches):
            randomized_batches.append(batch)
            randomized_tasks.append(task_num)
        task_num += 1
    print(
        "Time when all batches prepared for randomization is now:", datetime.now()
    )  # 12/15/22 added

    zipped = list(zip(randomized_batches, randomized_tasks))
    random.shuffle(zipped)
    print("Time when batches randomized is now:", datetime.now())  # 12/15/22 added
    randomized_batches, randomized_tasks = list(zip(*zipped))
    print(
        "Time when randomized batches converted to list now:", datetime.now()
    )  # 12/15/22 added

    return randomized_batches, randomized_tasks


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
):
    """
    Train_ds_list and val_ds_list are lists of MultTaskObject objects!
    Length of the list is the number of datasets used
    """
    num_tasks = len(datasets_list)

    print(f"Number of tasks: {num_tasks}")
    # get a list of the tasks by number
    for dset in datasets_list:
        train_state["tasks"].append(dset.task_num)
        train_state["train_avg_f1"][dset.task_num] = []
        train_state["val_avg_f1"][dset.task_num] = []

    # load data here -- only once (but Data should be shuffled each time)
    train_data, _ = get_data_from_loader(
        datasets_list,
        batch_size,
        shuffle=True,
        device=device,
        partition="train",
        sampler=sampler,
    )

    # load data here -- only once (but Data should be shuffled each time)
    dev_data, _ = get_data_from_loader(
        datasets_list,
        batch_size,
        shuffle=True,
        device=device,
        partition="dev",
        sampler=sampler,
    )

    for epoch_index in range(num_epochs):
        first = datetime.now()
        print(f"Starting epoch {epoch_index} at {first}")

        train_state["epoch_index"] = epoch_index

        # get running loss, holders of ys and predictions on training partition
        running_loss, ys_holder, preds_holder = run_model(
            datasets_list,  # todo: refactor to remove me
            train_data,
            classifier,
            batch_size,
            num_tasks,
            device,
            use_speaker,
            use_gender,
            avgd_acoustic,
            optimizer,
            train_state,
            mode="training",
            save_encoded_data=save_encoded_data,
            loss_fx=loss_fx,
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
            dev_data,
            classifier,
            batch_size,
            num_tasks,
            device,
            use_speaker,
            use_gender,
            avgd_acoustic,
            optimizer,
            train_state,
            mode="eval",
            save_encoded_data=save_encoded_data,
            loss_fx=loss_fx,
        )

        # get precision, recall, f1 info
        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="weighted"
            )
            print(f"Val weighted f-score for task {task}: {task_avg_f1}")
            # add val f1 to train state
            train_state["val_avg_f1"][task].append(task_avg_f1[2])

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
        train_state = update_train_state(model=classifier, train_state=train_state)

        # update scheduler if there is one
        if scheduler is not None:
            scheduler.step(train_state["val_loss"][-1])

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break

        # print out how long this epoch took
        last = datetime.now()
        print(f"Epoch {epoch_index} completed at {last}")
        print(f"This epoch took {last - first}")
        sys.stdout.flush()


def run_model(
    datasets_list,
    partition_data,
    classifier,
    batch_size,
    num_tasks,
    device,
    use_speaker,
    use_gender,
    avgd_acoustic,
    optimizer,
    train_state,
    mode="training",
    save_encoded_data=False,
    loss_fx=None,
):
    """
    Run the model in either training or testing within a single epoch
    Returns running_loss, gold labels, and predictions
    """
    first = datetime.now()

    # Iterate over training dataset
    running_loss = 0.0

    # get batches and tasks
    batches, tasks = randomize_batches(partition_data)

    next_time = datetime.now()
    print(f"Batches organized at {next_time - first}")

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
            batch_index,
            batch_task,
            classifier,
            device,
            use_speaker,
            use_gender,
            avgd_acoustic,
            # save_encoded_data=save_encoded_data #todo: add me
        )

        # print("batch_task is:", batch_task) #added 12/06/22
        # print("batch_pred is:", batch_pred) #added 12/06/22
        # print("y_gold is:", y_gold) #added 12/06/22
        # print(batch_pred.size()) #added 12/06/22

        # calculate loss
        if loss_fx:
            loss = (
                loss_fx(batch_pred, y_gold) * datasets_list[batch_task].loss_multiplier
            )
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

    then_time = datetime.now()
    print(f"Train set finished for epoch at {then_time - next_time}")

    return running_loss, ys_holder, preds_holder


def get_predictions(
    batch,
    batch_index,
    batch_task,
    classifier,
    device,
    use_speaker,
    use_gender,
    avgd_acoustic,
    save_encoded_data=False,
):
    """
    Get predictions from data
    This should abstract train and dev into a single function
    Used with multitask and prototypical networks so far
    """
    # get parts of batches
    # get data
    # todo add flexibilty for other tasks in same dataset
    y_gold = batch["ys"][0].detach().to(device)

    batch_text = batch["x_utt"].detach().to(device)
    # print(batch_text) # added 11/29/22
    # print(type(batch_text)) # added 11/29/22
    # print(batch_text.size()) # added 11/29/22
    if use_speaker:
        batch_speakers = batch["x_speaker"].to(device)
    else:
        batch_speakers = None

    if use_gender:
        batch_genders = batch["x_gender"].to(device)
    else:
        batch_genders = None
    batch_lengths = batch["utt_length"]

    # feed these parts into classifier
    # compute the output
    if avgd_acoustic:
        y_pred = classifier(
            text_input=batch_text,
            speaker_input=batch_speakers,
            length_input=batch_lengths,
            gender_input=batch_genders,
            # task_num=tasks[batch_index],
            save_encoded_data=save_encoded_data,
        )
    else:
        y_pred = classifier(
            text_input=batch_text,
            speaker_input=batch_speakers,
            length_input=batch_lengths,
            gender_input=batch_genders,
            # task_num=tasks[batch_index],
            save_encoded_data=save_encoded_data,
        )

    batch_pred = y_pred
    # print("y predictions are:")
    # print(y_pred)
    # print(batch_task)
    # print("Now printing batch predictions from line 375 of train_and_test_single_models.py")
    # print(batch_pred)

    # todo: make sure we can remove this
    # if datasets_list[batch_task].binary:
    #     batch_pred = batch_pred.float()
    #     y_gold = y_gold.float()

    return y_gold, batch_pred


def make_train_state(learning_rate, model_save_file, early_stopping_criterion):
    # makes a train state to save information on model during training/testing
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
