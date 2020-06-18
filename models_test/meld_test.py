# test the models created in py_code directory
# currently the main entry point into the system

from models.bimodal_models import BimodalCNN, MultichannelCNN
from models.baselines import LRBaseline, EmbeddingsOnly
from models.train_and_test_models import *

from models.input_models import *
from data_prep.data_prep import *
from data_prep.meld_input_formatting import *

# import parameters for model
# comment or uncomment as needed
from models.parameters.multitask_params import params

import numpy as np
import random
import torch
import sys


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
glove_file = "../../glove.short.300d.txt"  # should be updated later to a glove subset appropriate for this task

meld_path = "../../MELD_formatted"
# set number of splits
num_splits = params.num_splits
# set model name and model type
model = params.model
model_type = "Multitask_k=4"
# set number of columns to skip in data input files
cols_to_skip = params.cols_to_skip
# path to directory where best models are saved
model_save_path = "output/models/"
# decide if you want to plot the loss/accuracy curves for training
get_plot = True


if __name__ == "__main__":
    # 1. IMPORT AUDIO AND TEXT
    #
    # # make acoustic dict
    # # uncomment use_cols=... to use only specific columns from input data
    # acoustic_dict = make_acoustic_dict(input_dir, "_IS10_avgd.csv") #,
    #                                    # use_cols=['word', 'speaker', 'utt_num', 'word_num',
    #                                    #           'pcm_loudness_sma', 'F0final_sma', 'jitterLocal_sma',
    #                                    #           'shimmerLocal_sma', 'pcm_loudness_sma_de',
    #                                    #           'F0final_sma_de', 'jitterLocal_sma_de',
    #                                    #           'shimmerLocal_sma_de'])
    # print("Acoustic dict created")

    # 2. IMPORT GLOVE + MAKE GLOVE OBJECT
    glove_dict = make_glove_dict(glove_file)
    glove = Glove(glove_dict)
    print("Glove object created")

    # 3. MAKE DATASET
    data = MELDData(meld_path=meld_path, glove=glove, acoustic_length=params.audio_dim)
    print("Dataset created")

    # 6. CREATE NN
    # get set of pretrained embeddings and their shape
    pretrained_embeddings = data.glove.data
    num_embeddings = pretrained_embeddings.size()[0]
    print("shape of pretrained embeddings is: {0}".format(data.glove.data.size()))
    # num_embeddings = None
    # pretrained_embeddings = None

    # get the number of utterances per data point
    # todo: will need to change to variable number of utts per point
    num_utts = len(data.test_utts[0])  # data format should be [DIALOGUES][UTTS][WDS]
    # num_utts = data.x_acoustic.shape[1]
    # print(num_utts)

    # prepare holders for loss and accuracy of best model versions
    all_test_losses = []
    all_test_accs = []

    # mini search through different learning_rate values
    for lr in params.lrs:

        # instantiate empty model holder
        bimodal_predictor = None

        # this uses train-dev-test folds
        # create instance of model
        if params.model == "LRBaseline":
            # cleanup needed for baselines
            bimodal_trial = LRBaseline(params=params, num_embeddings=num_embeddings,
                                       pretrained_embeddings=pretrained_embeddings)
        elif params.model == "MultichannelCNN":
            bimodal_trial = MultichannelCNN(params=params, num_embeddings=num_embeddings,
                                            pretrained_embeddings=pretrained_embeddings)
        elif params.model == "EmbeddingsOnly":
            bimodal_trial = EmbeddingsOnly(params=params, num_embeddings=num_embeddings,
                                           pretrained_embeddings=pretrained_embeddings)
        elif params.model == "Multitask":
            bimodal_trial = BasicEncoder(params=params, num_embeddings=num_embeddings,
                                         pretrained_embeddings=pretrained_embeddings)
            bimodal_predictor = EmotionToSuccessFFNN(params=params, num_utts=num_utts,
                                                     num_layers=2, hidden_dim=4,
                                                     output_dim=1)
        elif params.model == "Multitask-meld":
            bimodal_trial =  BasicEncoder(params=params, num_embeddings=num_embeddings,
                                          pretrained_embeddings=pretrained_embeddings)
        else:
            # default to bimodal cnn
            bimodal_trial = BimodalCNN(params=params, num_embeddings=num_embeddings,
                                       pretrained_embeddings=pretrained_embeddings)

        # set the classifier(s) to the right device
        bimodal_trial = bimodal_trial.to(device)
        if bimodal_predictor is not None:
            bimodal_predictor.to(device)

        # set loss function, optimization, and scheduler, if using
        loss_func = nn.BCELoss()
        # loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(lr=lr, params=bimodal_trial.parameters(),
                                     weight_decay=params.weight_decay)
        if params.use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                   mode='min', factor=params.scheduler_factor,
                                                                   patience=params.scheduler_patience)

        print("Model, loss function, and optimization created")

        train_data = data.train_data
        print(type(train_data))
        # sys.exit(1)
        dev_data = data.dev_data
        test_data = data.test_data

        # create a a save path and file for the model
        model_save_file = "{0}_batch{1}_{2}hidden_{3}lyrs_lr{4}_{5}batch.pth".format(
            model_type, params.batch_size, params.hidden_dim, params.num_layers, lr,
            params.batch_size
        )

        # create 2 save paths for models if using both encoder + decoder
        if params.model == "Multitask":
            # model_save_file = "{0}_batch{1}_{2}hidden_{3}lyrs_lr{4}_{5}batch_encoder.pth".format(
            #     model_type, params.batch_size, params.hidden_dim, params.num_layers, lr,
            #     params.batch_size
            # )
            model2_save_file = "{0}__batch{1}_{2}hidden_{3}lyrs_lr{4}_{5}batch_decoder.pth".format(
                model_type, params.batch_size, params.hidden_dim, params.num_layers, lr,
                params.batch_size
            )

            train_state_2 = make_train_state(lr, model_save_path, model2_save_file)
            load_path2 = model_save_path + model2_save_file

        # make the train state to keep track of model training/development
        train_state = make_train_state(lr, model_save_path, model_save_file)

        load_path = model_save_path + model_save_file

        # train the model and evaluate on development set
        if params.model == "Multitask":
            train_and_predict(bimodal_trial, train_state, train_data, dev_data, params.batch_size,
                              params.num_epochs, loss_func, optimizer, device, scheduler=None,
                              model2=bimodal_predictor, train_state2=train_state_2)
        else:
            train_and_predict(bimodal_trial, train_state, train_data, dev_data, params.batch_size,
                              params.num_epochs, loss_func, optimizer, device, scheduler=None)

        # plot the loss and accuracy curves
        if get_plot:
            if params.model == "Multitask":
                plot_train_dev_curve(train_state_2['train_loss'], train_state_2['val_loss'], x_label="Epoch",
                                     y_label="Loss",
                                     title="Training and Dev loss for normed model {0} with lr {1}".format(
                                         model_type, lr),
                                     save_name="output/plots/{0}_lr{1}_loss.png".format(model_type, lr),
                                     set_axis_boundaries=False)
                # plot the accuracy
                plot_train_dev_curve(train_state_2['train_acc'], train_state_2['val_acc'], x_label="Epoch",
                                     y_label="Accuracy",
                                     title="Training and Dev accuracy for normed model {0} with lr {1}".format(
                                         model_type, lr),
                                     save_name="output/plots/{0}_lr{1}_acc.png".format(model_type, lr), losses=False,
                                     set_axis_boundaries=False)
            else:
            # loss curve
                plot_train_dev_curve(train_state['train_loss'], train_state['val_loss'], x_label="Epoch",
                                     y_label="Loss",
                                     title="Training and Dev loss for normed model {0} with lr {1}".format(
                                         model_type, lr),
                                     save_name="output/plots/{0}_lr{1}_loss.png".format(model_type, lr),
                                     set_axis_boundaries=False)
                # plot the accuracy
                plot_train_dev_curve(train_state['train_acc'], train_state['val_acc'], x_label="Epoch",
                                     y_label="Accuracy",
                                     title="Training and Dev accuracy for normed model {0} with lr {1}".format(
                                         model_type, lr),
                                     save_name="output/plots/{0}_lr{1}_acc.png".format(model_type, lr), losses=False,
                                     set_axis_boundaries=False)

        # add best evaluation losses and accuracy from training to set
        all_test_losses.append(train_state['early_stopping_best_val'])
        all_test_accs.append(train_state['best_val_acc'])

    # print the best model losses and accuracies for each development set in the cross-validation
    for i, item in enumerate(all_test_losses):
        print("Losses for model with lr={0}: {1}".format(params.lrs[i], item))
        print("Accuracy for model with lr={0}: {1}".format(params.lrs[i], all_test_accs[i]))
