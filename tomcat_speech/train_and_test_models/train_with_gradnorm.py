
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

from tomcat_speech.train_and_test_models.train_and_test_utils import update_train_state, get_all_batches, get_all_batches_oversampling

def multitask_train_and_predict_with_gradnorm(
    classifier,
    train_state,
    datasets_list,
    batch_size,
    num_epochs,
    optimizer1,
    device="cpu",
    avgd_acoustic=True,
    use_speaker=True,
    use_gender=False,
    optimizer2_learning_rate=0.001,
):
    """
    Train_ds_list and val_ds_list are lists of MultTaskObject objects!
    Length of the list is the number of datasets used
    Includes gradnorm from https://github.com/hosseinshn/GradNorm/blob/master/GradNormv10.ipynb
    """
    # set loss function
    loss_fx = nn.CrossEntropyLoss(reduction="mean")

    num_tasks = len(datasets_list)

    gradient_loss_fx = nn.L1Loss()

    # set holder for loss weights
    loss_weights = []

    # set holder for task loss 0s
    # todo: this is NOT how they do it in the gradnorm code
    #  but they only seem to have values for epoch 0...
    all_task_loss_0s = [0.0] * num_tasks

    # get a list of the tasks by number
    for dset in datasets_list:
        train_state["tasks"].append(dset.task_num)
        train_state["train_avg_f1"][dset.task_num] = []
        train_state["val_avg_f1"][dset.task_num] = []
        # initialize weight for each dataset
        loss_weights.append(torch.tensor(torch.FloatTensor([1]), requires_grad=True))

    optimizer2 = torch.optim.Adam(loss_weights, lr=optimizer2_learning_rate)

    for epoch_index in range(num_epochs):

        print("Now starting epoch {0}".format(epoch_index))

        train_state["epoch_index"] = epoch_index

        # Iterate over training dataset
        running_loss = 0.0

        # set classifier(s) to training mode
        classifier.train()

        batches, _ = get_all_batches_oversampling(
            datasets_list, batch_size=batch_size, shuffle=True
        )

        # print(f"printing length of all batches {len(batches)}")
        # set holders to use for error analysis
        ys_holder = {}
        for i in range(num_tasks):
            ys_holder[i] = []
        preds_holder = {}
        for i in range(num_tasks):
            preds_holder[i] = []

        # for each batch in the list of batches created by the dataloader
        for batch_index, batch in enumerate(batches):
            # set holder for all task losses and loss weights for the batch
            all_task_losses = []

            # go through each task in turn from within the batch
            for task_idx, task_batch in enumerate(batch):
                # identify the task for this portion of the batch
                batch_task = task_idx
                # get gold labels from the task
                y_gold = task_batch[4].to(device)

                batch_acoustic = task_batch[0].to(device)
                batch_text = task_batch[1].to(device)
                if use_speaker:
                    batch_speakers = task_batch[2].to(device)
                else:
                    batch_speakers = None

                if use_gender:
                    batch_genders = task_batch[3].to(device)
                else:
                    batch_genders = None
                batch_lengths = task_batch[-2].to(device)
                batch_acoustic_lengths = task_batch[-1].to(device)

                if avgd_acoustic:
                    y_pred = classifier(
                        acoustic_input=batch_acoustic,
                        text_input=batch_text,
                        speaker_input=batch_speakers,
                        length_input=batch_lengths,
                        gender_input=batch_genders,
                        task_num=batch_task,
                    )
                else:
                    y_pred = classifier(
                        acoustic_input=batch_acoustic,
                        text_input=batch_text,
                        speaker_input=batch_speakers,
                        length_input=batch_lengths,
                        acoustic_len_input=batch_acoustic_lengths,
                        gender_input=batch_genders,
                        task_num=batch_task,
                    )

                batch_pred = y_pred[batch_task]

                # get the loss for that task in that batch
                task_loss = loss_weights[batch_task] * loss_fx(batch_pred, y_gold)
                all_task_losses.append(task_loss)

                # for first epoch, set loss per item
                # todo: try log(3) as in footnote 2 of paper
                if epoch_index == 0:
                    task_loss_0 = task_loss.item()
                    all_task_loss_0s[task_idx] = task_loss_0

                # add ys to holder for error analysis
                preds_holder[batch_task].extend(
                    [item.index(max(item)) for item in batch_pred.tolist()]
                )
                ys_holder[batch_task].extend(y_gold.tolist())

            # calculate total loss
            loss = torch.div(sum(all_task_losses), len(all_task_losses))

            optimizer1.zero_grad()

            # use loss to produce gradients
            loss.backward(retain_graph=True)

            # get gradients of first layer of task-specific calculations
            param = list(classifier.parameters())
            final_shared_lyr_wt = param[40]
            all_normed_grads = []
            for task in range(num_tasks):
                # use the final shared layer weights to calculate gradient
                # here, this is param[40]
                task_grad = torch.autograd.grad(
                    all_task_losses[task], final_shared_lyr_wt, create_graph=True
                )
                normed_grad = torch.norm(task_grad[0], 2)
                all_normed_grads.append(normed_grad)

            # calculate average of normed gradients
            normed_grad_avg = torch.div(sum(all_normed_grads), len(all_normed_grads))

            # calculate relative losses
            all_task_loss_hats = []

            for task in range(num_tasks):
                task_loss_hat = torch.div(all_task_losses[task], all_task_loss_0s[task])
                all_task_loss_hats.append(task_loss_hat)
            loss_hat_avg = torch.div(sum(all_task_loss_hats), len(all_task_loss_hats))

            # calculate relative inverse training rate for tasks
            all_task_inv_rates = []
            for task in range(num_tasks):
                task_inv_rate = torch.div(all_task_loss_hats[task], loss_hat_avg)
                all_task_inv_rates.append(task_inv_rate)

            # calculate constant target for gradnorm paper equation 2
            alph = 0.16  # as selected in paper. could move to config + alter
            all_C_values = []
            for task in range(num_tasks):
                task_C = normed_grad_avg * all_task_inv_rates[task] ** alph
                task_C = task_C.detach()
                all_C_values.append(task_C)

            optimizer2.zero_grad()

            # calculate gradient loss using equation 2 in gradnorm paper
            all_task_gradient_losses = []
            for task in range(num_tasks):
                task_gradient_loss = gradient_loss_fx(
                    all_normed_grads[task], all_C_values[task]
                )
                all_task_gradient_losses.append(task_gradient_loss)
            gradient_loss = sum(all_task_gradient_losses)
            # propagate the loss
            gradient_loss.backward(retain_graph=True)

            # increment weights for loss
            optimizer2.step()

            # increment optimizer for model
            optimizer1.step()

            # renormalize the loss weights
            all_weights = sum(loss_weights)
            coef = num_tasks / all_weights
            loss_weights = [item * coef for item in loss_weights]

            # get loss calculations for train state
            # this is NOT gradnorm's calculation
            loss_t = loss.item()

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

        # add loss and accuracy information to the train state
        train_state["train_loss"].append(running_loss)

        for task in preds_holder.keys():
            task_avg_f1 = precision_recall_fscore_support(
                ys_holder[task], preds_holder[task], average="weighted"
            )
            print(f"Training weighted f-score for task {task}: {task_avg_f1}")
            train_state["train_avg_f1"][task].append(task_avg_f1[2])

        # Iterate over validation set--put it in a dataloader
        batches, tasks = get_all_batches(
            datasets_list, batch_size=batch_size, shuffle=True, partition="dev"
        )

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

        # holder for losses for each task in dev set
        task_val_losses = []
        for _ in range(num_tasks):
            task_val_losses.append(0.0)

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
                    task_num=tasks[batch_index],
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
                )

            batch_pred = y_pred[batch_task]

            if datasets_list[batch_task].binary:
                batch_pred = batch_pred.float()
                y_gold = y_gold.float()

            # calculate loss
            loss = loss_weights[batch_task] * loss_fx(batch_pred, y_gold)
            loss_t = loss.item()

            # calculate running loss
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # add ys to holder for error analysis
            preds_holder[batch_task].extend(
                [item.index(max(item)) for item in batch_pred.tolist()]
            )
            ys_holder[batch_task].extend(y_gold.tolist())

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

        # if it's time to stop, end the training process
        if train_state["stop_early"]:
            break
