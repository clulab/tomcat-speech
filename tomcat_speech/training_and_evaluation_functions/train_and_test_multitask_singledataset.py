
from datetime import datetime

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from tomcat_speech.training_and_evaluation_functions.train_and_test_utils import (
    get_all_batches,
    update_train_state
)


def train_and_predict_multitask_singledataset(
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
        Train_ds_list and val_ds_list are lists of MultTaskObject objects!
        Length of the list is the number of datasets used
        """
    num_tasks = 3

    print(f"Number of tasks: {num_tasks}")
    # get a list of the tasks by number
    for task in range(num_tasks):
        train_state["tasks"].append(task)
        train_state["train_avg_f1"][task] = []
        train_state["val_avg_f1"][task] = []
        train_state["val_best_f1"].append(0)

    for epoch_index in range(num_epochs):

        first = datetime.now()
        print(f"Starting epoch {epoch_index} at {first}")

        train_state["epoch_index"] = epoch_index

        # get running loss, holders of ys and predictions on training partition
        running_loss, ys_holder, preds_holder = run_model_multitask_singledataset(
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
            save_encoded_data=save_encoded_data,
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
        running_loss, ys_holder, preds_holder = run_model_multitask_singledataset(
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
            save_encoded_data=save_encoded_data,
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
        print(f"Epoch {epoch_index} completed at {last}")
        print(f"This epoch took {last - first}")
        sys.stdout.flush()


def run_model_multitask_singledataset(
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
    save_encoded_data=False,
    loss_fx=None,
    sampler=None,
    use_spec=False
):
    """
    Run the model in either training or testing within a single epoch
    Returns running_loss, gold labels, and predictions
    """
    batch_task = None

    first = datetime.now()

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

        # zero gradients
        if mode.lower() == "training" or mode.lower() == "train":
            optimizer.zero_grad()

        # get ys and predictions for the batch
        y_gold, batch_pred = get_asist_predictions(
            batch,
            batch_index,
            batch_task,
            classifier,
            device,
            use_speaker,
            use_gender,
            avgd_acoustic,
            tasks,
            datasets_list,
            use_spec=use_spec,
            #save_encoded_data=save_encoded_data #todo: add me
        )

        # calculate loss
        for task, preds in enumerate(batch_pred):
            batch_losses = []
            if loss_fx:
                loss = loss_fx[task](preds, y_gold[task]) * datasets_list[0].loss_multiplier
            else:
                loss = (
                    datasets_list[0].loss_fx[task](preds, y_gold[task])
                    * datasets_list[0].loss_multiplier
                )
                batch_losses.append(loss)

            loss_t = loss.item()

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # use loss to produce gradients
            if mode.lower() == "training" or mode.lower() == "train":
                loss.backward(retain_graph=True)

            # add ys to holder for error analysis
            preds_holder[task].extend(
                [item.index(max(item)) for item in preds.detach().tolist()]
            )
            ys_holder[task].extend(y_gold[task].detach().tolist())

        # increment optimizer
        if mode.lower() == "training" or mode.lower() == "train":
            optimizer.step()

    then_time = datetime.now()
    print(f"Train set finished for epoch at {then_time - next_time}")

    return running_loss, ys_holder, preds_holder


def get_asist_predictions(
    batch,
    batch_index,
    batch_task,
    classifier,
    device,
    use_speaker,
    use_gender,
    avgd_acoustic,
    tasks,
    datasets_list,
    use_spec=False,
    save_encoded_data=False
):
    """
    Get predictions from asist data ON THREE TASKS
    This should abstract train and dev into a single function
    Used with multitask networks
    """
    # get parts of batches
    # get data
    for item in batch["ys"]:
        item = item.detach().to(device)
    y_gold = batch["ys"]

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
    batch_lengths = batch["utt_length"] # .to(device)
    batch_acoustic_lengths = batch["acoustic_length"] # .to(device)
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
            task_num=tasks[batch_index],
            save_encoded_data=save_encoded_data
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
            task_num=tasks[batch_index],
            save_encoded_data=save_encoded_data
        )

    batch_pred = y_pred[:3]

    return y_gold, batch_pred
