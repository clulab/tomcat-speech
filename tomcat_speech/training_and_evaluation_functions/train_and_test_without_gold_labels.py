# perform multitask prediction on data that does not contain gold labels
import pickle
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

from tomcat_speech.training_and_evaluation_functions.train_and_test_utils import (
    get_all_batches,
)


def multitask_predict_without_gold_labels(
    classifier,
    test_ds,
    batch_size,
    device="cpu",
    num_predictions=2,
    avgd_acoustic=True,
    use_speaker=True,
    use_gender=False,
    get_prob_dist=False,
    return_penultimate_layer=False,
    use_spec=False,
):
    """
    Test a pretrained model
    """
    # get number of tasks
    num_tasks = num_predictions

    # ordered ids
    ordered_ids = []

    # get holder for predictions
    preds_holder = {}
    for i in range(num_tasks):
        preds_holder[i] = []

    if return_penultimate_layer:
        penult_holder = {}
        for i in range(num_tasks):
            penult_holder[i] = []

    # Iterate over validation set--put it in a dataloader
    test_batches = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # set classifier to evaluation mode
    classifier.eval()

    # for each batch in the dataloader
    for batch_index, batch in enumerate(test_batches):
        if type(batch) == list:
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
        else:
            batch_acoustic = batch["x_acoustic"].to(device)
            batch_text = batch["x_utt"].to(device)
            batch_lengths = batch["utt_length"].to(device)
            batch_acoustic_lengths = batch["acoustic_length"].to(device)
            if use_speaker:
                batch_speakers = batch["x_speaker"].to(device)
            else:
                batch_speakers = None

            if use_gender:
                batch_genders = batch["x_gender"].to(device)
            else:
                batch_genders = None

            if use_spec:
                batch_spec = batch["x_spec"].to(device)
            else:
                batch_spec = None

            ordered_ids.extend(batch["audio_id"])

        if avgd_acoustic:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                spec_input=batch_spec,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                gender_input=batch_genders,
                get_prob_dist=get_prob_dist,
                return_penultimate_layer=return_penultimate_layer,
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
                get_prob_dist=get_prob_dist,
                return_penultimate_layer=return_penultimate_layer,
            )

        # separate predictions and penultimate layers if needed
        # add penultimates to a penult holder equivalent to preds_holder
        if return_penultimate_layer:
            penults = [item[1] for item in y_pred]
            y_pred = [item[0] for item in y_pred]

            for i in range(num_tasks):
                penult_holder[i].extend([penults[i].tolist()])

        if get_prob_dist:
            for i in range(num_tasks):
                preds_holder[i].extend([y_pred[i].tolist()])
        else:
            # add ys to holder for error analysis
            for i in range(num_tasks):
                preds_holder[i].extend(
                    [(item.index(max(item)), max(item)) for item in y_pred[i].tolist()]
                )

    if not return_penultimate_layer:
        return preds_holder, ordered_ids
    else:
        return preds_holder, penult_holder, ordered_ids


# unused?
# GENERATES LIST OF LISTS WITH PREDICTION AND CONFIDENCE LEVELS
def predict_without_gold_labels(
    classifier,
    test_ds,
    batch_size,
    device="cpu",
    avgd_acoustic=True,
    use_speaker=True,
    use_gender=False,
    get_prob_dist=False,
):
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
                get_prob_dist=get_prob_dist,
            )
        else:
            y_pred = classifier(
                acoustic_input=batch_acoustic,
                text_input=batch_text,
                speaker_input=batch_speakers,
                length_input=batch_lengths,
                acoustic_len_input=batch_acoustic_lengths,
                gender_input=batch_genders,
                get_prob_dist=get_prob_dist,
            )

        # add ys to holder for error analysis
        preds_holder.extend(
            [[item.index(max(item)), max(item)] for item in y_pred.tolist()]
        )

    return preds_holder


# unused?
def single_dataset_multitask_predict(
    classifier,
    train_state,
    datasets_list,
    batch_size,
    pickle_save_name,
    device="cpu",
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

    # for each batch in the list of batches created by the dataloader
    for batch_index, batch in enumerate(batches):
        # get the task for this batch
        batch_task = tasks[batch_index]

        y_gold = batch[4].to(device)

        batch_ids = batch[-3]
        ids_holder[batch_task].extend(batch_ids)

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
