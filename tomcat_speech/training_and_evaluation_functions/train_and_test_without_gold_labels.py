# perform multitask prediction on data that does not contain gold labels
from torch.utils.data import DataLoader


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
    Test a pretrained model without gold labels
    :param classifier: a trained pytorch model used for classification
    :param test_ds: the test data used
    :param batch_size: the size of minibatches
    :param device: 'cpu' or 'cuda'
    :param num_predictions: the number of tasks per item
    :param avgd_acoustic: whether acoustic features have been averaged
    :param use_speaker: whether speaker embeddings are used
    :param use_gender: whether speaker gender embeddings are used
    :param get_prob_dist: whether to return a probability distribution
        over the label space for each prediction made
    :param return_penultimate_layer: whether to return the penultimate
        hidden layer of the model
    :param use_spec: whether to use spectrograms
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
