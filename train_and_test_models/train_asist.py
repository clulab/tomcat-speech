# test the models created in py_code directory with ASIST dataset
# currently the main entry point into the system
# add "prep_data" as an argument when running this from command line
#       if your acoustic features have not been extracted from audio

from models.bimodal_models import MultichannelCNN
from models.baselines import LRBaseline
from models.train_and_test_models import *
from models.input_models import *

from data_prep.data_prep_helpers import *
from data_prep.lives_data.lives_prep import make_acoustic_dict, ClinicalDataset
import data_prep.asist_data.sentiment_score_prep as score_prep
import data_prep.asist_data.asist_prep as asist_prep

# import parameters for model
# comment or uncomment as needed
from models.parameters.bimodal_params import params

# from models.parameters.multitask_params import params
# from models.parameters.lr_baseline_1_params import params
# from models.parameters.multichannel_cnn_params import params

import numpy as np
import random
import torch
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
seed = params.seed
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
num_splits = params.num_splits
# set model name and model type
model = params.model
model_type = "BimodalCNN_k=4"
# set number of columns to skip in data input files
cols_to_skip = params.cols_to_skip
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
    data = ClinicalDataset(
        acoustic_dict,
        glove,
        cols_to_skip=cols_to_skip,
        ys_path=y_path,
        splits=num_splits,
        sequence_prep="pad",
        truncate_from="start",
        alignment=params.alignment,
    )
    print("Dataset created")

    # 6. CREATE NN
    # get set of pretrained embeddings and their shape
    pretrained_embeddings = data.glove.data
    num_embeddings = pretrained_embeddings.size()[0]
    print("shape of pretrained embeddings is: {0}".format(data.glove.data.size()))

    # get the number of utterances per data point
    num_utts = data.x_acoustic.shape[1]

    # prepare holders for loss and accuracy of best model versions
    all_test_losses = []
    all_test_accs = []

    # mini search through different learning_rate values
    for lr in params.lrs:

        # prep intermediate loss and acc holders
        # feed these into all_test_losses/accs
        all_y_acc = []
        all_y_loss = []

        # use each train-val=test split in a separate training routine
        for split in range(data.splits):
            print("Now starting training/tuning with split {0} held out".format(split))

            # instantiate empty model holder
            bimodal_predictor = None

            # create instance of model
            if params.model == "LRBaseline":
                # cleanup needed for baselines
                bimodal_trial = LRBaseline(
                    params=params,
                    num_embeddings=num_embeddings,
                    pretrained_embeddings=pretrained_embeddings,
                )
            elif params.model == "MultichannelCNN":
                bimodal_trial = MultichannelCNN(
                    params=params,
                    num_embeddings=num_embeddings,
                    pretrained_embeddings=pretrained_embeddings,
                )
            elif params.model == "EmbeddingsOnly":
                bimodal_trial = EmbeddingsOnly(
                    params=params,
                    num_embeddings=num_embeddings,
                    pretrained_embeddings=pretrained_embeddings,
                )
            elif params.model == "Multitask":
                bimodal_trial = BasicEncoder(
                    params=params,
                    num_embeddings=num_embeddings,
                    pretrained_embeddings=pretrained_embeddings,
                )
                bimodal_predictor = EmotionToSuccessFFNN(
                    params=params,
                    num_utts=num_utts,
                    num_layers=2,
                    hidden_dim=4,
                    output_dim=1,
                )
            else:
                # default to bimodal cnn
                bimodal_trial = BimodalCNN(
                    params=params,
                    num_embeddings=num_embeddings,
                    pretrained_embeddings=pretrained_embeddings,
                )

            # set the classifier(s) to the right device
            bimodal_trial = bimodal_trial.to(device)
            if bimodal_predictor is not None:
                bimodal_predictor.to(device)

            # set loss function, optimization, and scheduler, if using
            loss_func = nn.BCELoss()
            # loss_func = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(
                lr=lr,
                params=bimodal_trial.parameters(),
                weight_decay=params.weight_decay,
            )
            if params.use_scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    mode="min",
                    factor=params.scheduler_factor,
                    patience=params.scheduler_patience,
                )

            print("Model, loss function, and optimization created")

            # set train-val-test splits
            # the number in 'split' becomes the test/holdout split
            data.set_split(split)
            holdout = data.current_split
            val_split = data.val_split
            training_data = data.remaining_splits

            # create a a save path and file for the model
            model_save_file = "{0}_{1}_batch{2}_{3}hidden_{4}lyrs_lr{5}_{6}batch.pth".format(
                model_type,
                split,
                params.batch_size,
                params.hidden_dim,
                params.num_layers,
                lr,
                params.batch_size,
            )

            # create 2 save paths for models if using both encoder + decoder
            if params.model == "Multitask":
                model_save_file = "{0}_{1}_batch{2}_{3}hidden_{4}lyrs_lr{5}_{6}batch_encoder.pth".format(
                    model_type,
                    split,
                    params.batch_size,
                    params.hidden_dim,
                    params.num_layers,
                    lr,
                    params.batch_size,
                )
                model2_save_file = "{0}_{1}_batch{2}_{3}hidden_{4}lyrs_lr{5}_{6}batch_decoder.pth".format(
                    model_type,
                    split,
                    params.batch_size,
                    params.hidden_dim,
                    params.num_layers,
                    lr,
                    params.batch_size,
                )

                train_state_2 = make_train_state(lr, model_save_path, model2_save_file)
                load_path2 = model_save_path + model2_save_file

            # make the train state to keep track of model training/development
            train_state = make_train_state(lr, model_save_path, model_save_file)

            load_path = model_save_path + model_save_file

            # train the model and evaluate on development split
            if params.model == "Multitask":
                train_and_predict(
                    bimodal_trial,
                    train_state,
                    training_data,
                    val_split,
                    params.batch_size,
                    params.num_epochs,
                    loss_func,
                    optimizer,
                    device,
                    scheduler=None,
                    model2=bimodal_predictor,
                    train_state2=train_state_2,
                )
            else:
                train_and_predict(
                    bimodal_trial,
                    train_state,
                    training_data,
                    val_split,
                    params.batch_size,
                    params.num_epochs,
                    loss_func,
                    optimizer,
                    device,
                    scheduler=None,
                )

            # plot the loss and accuracy curves
            if get_plot:
                if params.model == "Multitask":
                    plot_train_dev_curve(
                        train_state_2["train_loss"],
                        train_state_2["val_loss"],
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
                        train_state_2["train_acc"],
                        train_state_2["val_acc"],
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
                else:
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
        print("Losses for model with lr={0}: {1}".format(params.lrs[i], item))
        print(
            "Accuracy for model with lr={0}: {1}".format(
                params.lrs[i], all_test_accs[i]
            )
        )
