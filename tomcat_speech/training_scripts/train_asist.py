# test the models created in py_code directory with ASIST dataset
# currently the main entry point into the system
# add "prep_data" as an argument when running this from command line
#       if your acoustic features have not been extracted from audio

from tomcat_speech.data_prep.asist_data.asist_dataset_creation import AsistDataset
from tomcat_speech.training_and_evaluation_functions.train_and_test_models import *
from tomcat_speech.models.plot_training import *
from tomcat_speech.models.multimodal_models import *
from tomcat_speech.training_and_evaluation_functions.train_and_test_utils import make_train_state

sys.path.append("../multimodal_data_preprocessing")
from utils.data_prep_helpers import (
    Glove,
    make_glove_dict,
)
from prep_data import make_acoustic_dict

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import tomcat_speech.data_prep.asist_data.sentiment_score_prep as score_prep
import tomcat_speech.data_prep.asist_data.asist_dataset_creation as asist_prep

# import parameters for model
# comment or uncomment as needed

# from tomcat_speech.models.parameters.multitask_params import params
# from tomcat_speech.models.parameters.lr_baseline_1_params import params
# from tomcat_speech.models.parameters.multichannel_cnn_params import params

import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import sys
import glob
import os

# set device
cuda = True

# Check CUDA
if not torch.cuda.is_available():
    cuda = False

device = torch.device("cuda" if cuda else "cpu")

# set random seed
seed = model_params.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)

# set parameters for data prep
glove_file = "../../glove.short.300d.txt"  # todo: change this?
input_dir = "output/asist_audio"
# to test the data--this doesn't contain real outcomes
y_path = "output/asist_audio/asist_ys/all_ys.csv"
# set number of splits
num_splits = 3
# set model name and model type
model = model_params.model
model_type = "BimodalCNN_k=4"
# set number of columns to skip in data input files
cols_to_skip = 5
# path to directory where best models are saved
model_save_path = "output/models/"
# make sure the full save path exists; if not, create it
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_save_path))
# decide if you want to plot the loss/accuracy curves for training
get_plot = True
model_plot_path = "output/plots/"
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_plot_path))

# set parameters for the sentiment analyzer prep
asist_transcription_path = "../../Downloads/data_flatstructure"
transcription_save_path = input_dir
sentiment_save_path = "output"
missions = ["mission_2"]
acoustic_feature_set = "IS10"
smile_path = "~/opensmile-2.3.0"


def asist_train_and_predict(
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
            # get gold labels; always at idx 4
            y_gold = batch[4].to(device)

            if split_point > 0:
                y_gold = torch.tensor(
                    [
                        1.0 if y_gold[i] > split_point else 0.0
                        for i in range(len(y_gold))
                    ]
                )

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

            # add ys to holder for error analysis
            if binary:
                preds_holder.extend([round(item[0]) for item in y_pred.tolist()])
            else:
                preds_holder.extend([item.index(max(item)) for item in y_pred.tolist()])
            ys_holder.extend(y_gold.tolist())

            # step 3. compute the loss
            loss = loss_func(y_pred, y_gold)
            loss_t = loss.item()  # loss for the item

            if len(list(y_pred.size())) > 1:
                if binary:
                    y_pred = torch.tensor([round(item[0]) for item in y_pred.tolist()])
                else:
                    y_pred = torch.tensor(
                        [item.index(max(item)) for item in y_pred.tolist()]
                    )
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
            y_gold = batch[4].to(device)

            if split_point > 0:
                y_gold = torch.tensor(
                    [
                        1.0 if y_gold[i] > split_point else 0.0
                        for i in range(len(y_gold))
                    ]
                )

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


if __name__ == "__main__":
    # 0. RUN ASIST DATA PREP AND REORGANIZATION FOR INPUT INTO THE MODEL
    if len(sys.argv) > 1 and sys.argv[1] == "prep_data":
        os.system("time python data_prep/asist_data/asist_prep.py")
    elif len(sys.argv) > 1 and sys.argv[1] == "mp4_data":
        os.system("time python data_prep/asist_data/asist_prep.py mp4_data")  # fixme
    # elif len(sys.argv) > 1 and sys.argv[1] == "use_sentiment_analyzer":
    #     os.system("time python data_prep/asist_data/asist_prep.py prep_for_sentiment_analyzer")

    # 1. IMPORT AUDIO AND TEXT
    # make acoustic dict
    acoustic_dict = make_acoustic_dict(input_dir, "_avgd.csv", data_type="asist")

    print("Acoustic dict created")

    # 1b. IMPORT SENTIMENT SCORES FOR INPUTS IF DESIRED
    if len(sys.argv) > 1 and sys.argv[1] == "use_sentiment_analyzer":
        sentiment_input = asist_prep.ASISTInput(
            asist_transcription_path,
            transcription_save_path,
            smile_path,
            missions=missions,
            acoustic_feature_set=acoustic_feature_set,
        )
        sentiment_score_files = asist_prep.run_sentiment_analysis_pipeline(
            sentiment_input, sentiment_save_path
        )

        print("Sentiment scores created and saved")

        # 1c. READ SENTIMENT SCORES INTO DICT AND GET THEM
        sentiment_utterance_files = glob.glob(
            sentiment_save_path + "/*_video_transcript_split.txt"
        )

        sent_scores = score_prep.SentimentScores(
            sentiment_save_path, sentiment_score_files
        )
        sent_scores.join_words_with_predictions(
            sentiment_save_path, sentiment_utterance_files
        )

        print("Sentiment scores and their corresponding utterances loaded")

        # sys.exit(1)

    # 2. IMPORT GLOVE + MAKE GLOVE OBJECT
    glove_dict = make_glove_dict(glove_file)
    glove = Glove(glove_dict)
    print("Glove object created")

    # 3. MAKE DATASET
    data = AsistDataset(
        acoustic_dict,
        glove,
        ys_path=y_path,
        splits=3,
        sequence_prep="pad",
        truncate_from="start",
        norm=None,
    )
    print("Dataset created")

    # 6. CREATE NN
    # get set of pretrained embeddings and their shape
    pretrained_embeddings = glove.data
    num_embeddings = pretrained_embeddings.size()[0]
    print(f"shape of pretrained embeddings is: {data.glove.data.size()}")

    # get the number of utterances per data point
    num_utts = data.x_acoustic.shape[1]

    # prepare holders for loss and accuracy of best model versions
    all_test_losses = []
    all_test_accs = []

    # mini search through different learning_rate values
    for lr in model_params.lrs:
        for wd in model_params.weight_decay:

            # prep intermediate loss and acc holders
            # feed these into all_test_losses/accs
            all_y_acc = []
            all_y_loss = []

            # use each train-val=test split in a separate training routine
            for split in range(data.splits):
                print(f"Now starting training/tuning with split {split} held out")

                # create instance of model
                bimodal_trial = IntermediateFusionMultimodalModel(
                    params=model_params,
                    num_embeddings=num_embeddings,
                    pretrained_embeddings=pretrained_embeddings,
                )

                # set loss function, optimization, and scheduler, if using
                loss_func = nn.BCELoss()
                # loss_func = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(
                    lr=lr, params=bimodal_trial.parameters(), weight_decay=wd,
                )

                print("Model, loss function, and optimization created")

                # set train-val-test splits
                # the number in 'split' becomes the test/holdout split
                data.set_split(split)
                holdout = data.current_split
                val_split = data.val_split
                training_data = data.remaining_splits

                # create a a save path and file for the model
                model_save_file = "{0}_batch{1}_{2}hidden_2lyrs_lr{3}.pth".format(
                    model_type, params.batch_size, params.fc_hidden_dim, lr
                )

                # make the train state to keep track of model training/development
                train_state = make_train_state(lr, model_save_path, model_save_file)

                load_path = model_save_path + model_save_file

                # train the model and evaluate on development split
                asist_train_and_predict(
                    bimodal_trial,
                    train_state,
                    training_data,
                    val_split,
                    model_params.batch_size,
                    model_params.num_epochs,
                    loss_func,
                    optimizer,
                    device,
                    use_speaker=model_params.use_speaker,
                    use_gender=model_params.use_gender,
                    scheduler=None,
                )

                # plot the loss and accuracy curves
                if get_plot:
                    # loss curve
                    plot_train_dev_curve(
                        train_state["train_loss"],
                        train_state["val_loss"],
                        x_label="Epoch",
                        y_label="Loss",
                        title="Training and Dev loss for normed model {0} split {1} with lr {2}".format(
                            model_type, split, lr
                        ),
                        save_name="output/plots/{0}_{1}_lr{2}_loss.png".format(
                            model_type, split, lr
                        ),
                    )
                    # plot the accuracy
                    plot_train_dev_curve(
                        train_state["train_acc"],
                        train_state["val_acc"],
                        x_label="Epoch",
                        y_label="Accuracy",
                        title="Training and Dev accuracy for normed model {0} split {1} with lr {2}".format(
                            model_type, split, lr
                        ),
                        save_name="output/plots/{0}_{1}_lr{2}_acc.png".format(
                            model_type, split, lr
                        ),
                        losses=False,
                    )

                # add best evaluation losses and accuracy from training to set
                all_y_loss.append(train_state["early_stopping_best_val"])
                all_y_acc.append(train_state["best_val_acc"])

            # print("Test loss on all folds: {0}".format(all_test_loss))
            # print("Test accuracy on all folds: {0}".format(all_test_acc))

            all_test_losses.append(all_y_loss)
            all_test_accs.append(all_y_acc)

    # print the best model losses and accuracies for each development set in the cross-validation
    for i, item in enumerate(all_test_losses):
        print("Losses for model with lr={0}: {1}".format(model_params.lrs[i], item))
        print(
            "Accuracy for model with lr={0}: {1}".format(
                model_params.lrs[i], all_test_accs[i]
            )
        )