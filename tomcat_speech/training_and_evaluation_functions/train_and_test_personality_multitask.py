
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from tomcat_speech.training_and_evaluation_functions.train_and_test_utils import update_train_state


def personality_as_multitask_train_and_predict(
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
    max_class=False,
):
    """
    Train for OCEAN personality traits
    Using each trait as a classification or regression task
    Currently MUST USE OCEANPersonalityModel as classifier
    """
    if max_class:
        num_tasks = 1
    else:
        num_tasks = 5

    # get a list of the tasks by number
    for num in range(num_tasks):
        train_state["tasks"].append(num)
        train_state["train_avg_f1"][num] = []
        train_state["val_avg_f1"][num] = []

    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # Iterate over training dataset
        running_loss = 0.0

        # set classifier(s) to training mode
        classifier.train()

        batches = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, sampler=sampler
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
            # step 1. zero the gradients
            optimizer.zero_grad()

            # step 2. compute the output
            if not max_class:
                gold_0 = batch[5].to(device)
                gold_1 = batch[6].to(device)
                gold_2 = batch[7].to(device)
                gold_3 = batch[8].to(device)
                gold_4 = batch[9].to(device)

                ys_holder[0].extend(gold_0.tolist())
                ys_holder[1].extend(gold_1.tolist())
                ys_holder[2].extend(gold_2.tolist())
                ys_holder[3].extend(gold_3.tolist())
                ys_holder[4].extend(gold_4.tolist())
            else:
                gold_0 = batch[4].to(device)
                ys_holder[0].extend(gold_0.tolist())

            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            batch_lengths = batch[-2].to(device)
            # get acoustic lengths if necessary
            if not avgd_acoustic:
                batch_acoustic_lengths = batch[-1].to(device)
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

            (
                trait_0_pred,
                trait_1_pred,
                trait_2_pred,
                trait_3_pred,
                trait_4_pred,
            ) = classifier(
                batch_acoustic,
                batch_text,
                batch_speakers,
                batch_lengths,
                batch_acoustic_lengths,
                batch_genders,
            )

            preds_holder[0].extend(
                [item.index(max(item)) for item in trait_0_pred.tolist()]
            )

            # step 3. compute the loss
            class_0_loss = loss_func(trait_0_pred, gold_0)

            if not max_class:
                preds_holder[1].extend(
                    [item.index(max(item)) for item in trait_1_pred.tolist()]
                )
                preds_holder[2].extend(
                    [item.index(max(item)) for item in trait_2_pred.tolist()]
                )
                preds_holder[3].extend(
                    [item.index(max(item)) for item in trait_3_pred.tolist()]
                )
                preds_holder[4].extend(
                    [item.index(max(item)) for item in trait_4_pred.tolist()]
                )

                class_1_loss = loss_func(trait_1_pred, gold_1)
                class_2_loss = loss_func(trait_2_pred, gold_2)
                class_3_loss = loss_func(trait_3_pred, gold_3)
                class_4_loss = loss_func(trait_4_pred, gold_4)

                loss = (
                    class_0_loss
                    + class_1_loss
                    + class_2_loss
                    + class_3_loss
                    + class_4_loss
                )
            else:
                loss = class_0_loss

            loss_t = loss.item()  # loss for the item

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            optimizer.step()

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)

        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="weighted"
            )
            print(f"Training weighted f-score for task {task}: {task_avg_f1}")
            train_state["train_avg_f1"][task].append(task_avg_f1[2])

        # Iterate over validation set--put it in a dataloader
        val_batches = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

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

        # for each batch in the dataloader
        # todo: what if there are different numbers of batches? (diff dataset sizes)
        for batch_index, batch in enumerate(val_batches):
            # step 2. compute the output
            if not max_class:
                gold_0 = batch[5].to(device)
                gold_1 = batch[6].to(device)
                gold_2 = batch[7].to(device)
                gold_3 = batch[8].to(device)
                gold_4 = batch[9].to(device)

                ys_holder[0].extend(gold_0.tolist())
                ys_holder[1].extend(gold_1.tolist())
                ys_holder[2].extend(gold_2.tolist())
                ys_holder[3].extend(gold_3.tolist())
                ys_holder[4].extend(gold_4.tolist())

            else:
                gold_0 = batch[4].to(device)
                ys_holder[0].extend(gold_0.tolist())

            batch_acoustic = batch[0].to(device)
            batch_text = batch[1].to(device)
            batch_lengths = batch[-2].to(device)
            # get acoustic lengths if necessary
            if not avgd_acoustic:
                batch_acoustic_lengths = batch[-1].to(device)
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

            (
                trait_0_pred,
                trait_1_pred,
                trait_2_pred,
                trait_3_pred,
                trait_4_pred,
            ) = classifier(
                batch_acoustic,
                batch_text,
                batch_speakers,
                batch_lengths,
                batch_acoustic_lengths,
                batch_genders,
            )

            preds_holder[0].extend(
                [item.index(max(item)) for item in trait_0_pred.tolist()]
            )

            # step 3. compute the loss
            class_0_loss = loss_func(trait_0_pred, gold_0)
            if not max_class:
                preds_holder[1].extend(
                    [item.index(max(item)) for item in trait_1_pred.tolist()]
                )
                preds_holder[2].extend(
                    [item.index(max(item)) for item in trait_2_pred.tolist()]
                )
                preds_holder[3].extend(
                    [item.index(max(item)) for item in trait_3_pred.tolist()]
                )
                preds_holder[4].extend(
                    [item.index(max(item)) for item in trait_4_pred.tolist()]
                )

                class_1_loss = loss_func(trait_1_pred, gold_1)
                class_2_loss = loss_func(trait_2_pred, gold_2)
                class_3_loss = loss_func(trait_3_pred, gold_3)
                class_4_loss = loss_func(trait_4_pred, gold_4)

                loss = (
                    class_0_loss
                    + class_1_loss
                    + class_2_loss
                    + class_3_loss
                    + class_4_loss
                )
            else:
                loss = class_0_loss

            # loss = loss_func(ys_pred, ys_gold)
            running_loss += (loss.item() - running_loss) / (batch_index + 1)

        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="weighted"
            )
            print(f"Val weighted f-score for task {task}: {task_avg_f1}")
            train_state["val_avg_f1"][task].append(task_avg_f1[2])

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
