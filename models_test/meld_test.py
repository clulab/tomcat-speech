# test the models created in models directory with MELD data
# currently the main entry point into the system

import numpy as np
import random
import torch
import sys
import pickle

sys.path.append("/net/kate/storage/work/bsharp/github/asist-speech")

from models.train_and_test_models import *

from models.input_models import *
from data_prep.data_prep import *
from data_prep.meld_input_formatting import *

# import parameters for model
from models.parameters.multitask_params import params

# set device
cuda = False
device = torch.device("cuda" if cuda else "cpu")

# set random seed
seed = params.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# set parameters for data prep
# todo: should be updated later to a glove subset appropriate for this task
# glove_file = "/work/bsharp/glove.short.300d.punct.txt"
# glove_file = "/data/nlp/corpora/glove/glove.840B.300d.no_proc_header.txt"
glove_file = "../../glove.short.300d.punct.txt"

# meld_path = "/data/nlp/corpora/MM/MELD_five_dialogues"
# meld_path = "/data/nlp/corpora/MM/MELD_formatted"
meld_path = "../../datasets/multimodal_datasets/MELD_formatted"

# set model name and model type
model = params.model
model_type = "DELETE_ME_FULL"
# path to directory where best models are saved
model_save_path = "output/models/"
# make sure the full save path exists; if not, create it
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_save_path))
# set dir to plot the loss/accuracy curves for training
model_plot_path = "output/plots/"
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_plot_path))


if __name__ == "__main__":

    # 1. IMPORT GLOVE + MAKE GLOVE OBJECT
    glove_dict = make_glove_dict(glove_file)
    glove = Glove(glove_dict)
    print("Glove object created")

    # 2. MAKE DATASET
    data = MELDData(meld_path=meld_path, glove=glove, acoustic_length=params.audio_dim)
    # with open('dataset_full', 'wb') as pickle_file:
    #     pickle.dump(data, pickle_file)
    # with open('dataset_full', 'rb') as pickle_file:
    #     data = pickle.load(pickle_file)

    data.emotion_weights = data.emotion_weights.to(device)  # add class weights to device
    print("Dataset created")

    # 3. CREATE NN
    # get set of pretrained embeddings and their shape
    pretrained_embeddings = data.glove.data
    num_embeddings = pretrained_embeddings.size()[0]
    print("shape of pretrained embeddings is: {0}".format(data.glove.data.size()))

    # prepare holders for loss and accuracy of best model versions
    all_test_losses = []
    all_test_accs = []

    # mini search through different learning_rate values
    for lr in params.lrs:

        # this uses train-dev-test folds
        # create instance of model
        bimodal_trial = BasicEncoder(params=params, num_embeddings=num_embeddings,
                                     pretrained_embeddings=pretrained_embeddings)

        # set the classifier(s) to the right device
        bimodal_trial = bimodal_trial.to(device)
        print(bimodal_trial)

        # set loss function, optimization, and scheduler, if using
        loss_func = nn.CrossEntropyLoss(reduction='mean')
        # loss_func = nn.CrossEntropyLoss(data.emotion_weights, reduction='mean')

        # optimizer = torch.optim.SGD(bimodal_trial.parameters(), lr=lr, momentum=0.9)
        optimizer = torch.optim.Adam(lr=lr, params=bimodal_trial.parameters(),
                                     weight_decay=params.weight_decay)

        print("Model, loss function, and optimization created")

        # set the train, dev, and set data
        train_data = data.train_data
        train_ds = DatumListDataset(train_data, data.emotion_weights)
        train_targets = torch.stack(list(train_ds.targets()))
        sampler_weights = data.emotion_weights
        train_samples_weights = sampler_weights[train_targets]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weights, len(train_samples_weights))

        dev_ds = DatumListDataset(data.dev_data, data.emotion_weights)
        test_ds = DatumListDataset(data.test_data, data.emotion_weights)

        # create a a save path and file for the model
        model_save_file = "{0}_batch{1}_{2}hidden_2lyrs_lr{3}.pth".format(
            model_type, params.batch_size, params.fc_hidden_dim, lr)

        # make the train state to keep track of model training/development
        train_state = make_train_state(lr, model_save_path, model_save_file)

        # set the load path for testing
        load_path = model_save_path + model_save_file

        # train the model and evaluate on development set
        train_and_predict(bimodal_trial, train_state, train_ds, dev_ds, params.batch_size,
                            params.num_epochs, loss_func, optimizer, device, scheduler=None, sampler=None)

        # plot the loss and accuracy curves
        # set plot titles
        loss_title = "Training and Dev loss for model {0} with lr {1}".format(model_type, lr)
        acc_title = "Training and Dev accuracy for model {0} with lr {1}".format(model_type, lr)

        # set save names
        loss_save = "output/plots/{0}_lr{1}_loss.png".format(model_type, lr)
        acc_save = "output/plots/{0}_lr{1}_acc.png".format(model_type, lr)

        # plot the loss from model
        plot_train_dev_curve(train_state['train_loss'], train_state['val_loss'], x_label="Epoch",
                                y_label="Loss", title=loss_title, save_name=loss_save,
                                set_axis_boundaries=False)
        # plot the accuracy from model
        plot_train_dev_curve(train_state['train_acc'], train_state['val_acc'], x_label="Epoch",
                                y_label="Accuracy", title=acc_title, save_name=acc_save, losses=False,
                                set_axis_boundaries=False)

        # add best evaluation losses and accuracy from training to set
        all_test_losses.append(train_state['early_stopping_best_val'])
        all_test_accs.append(train_state['best_val_acc'])

    # print the best model losses and accuracies for each development set in the cross-validation
    for i, item in enumerate(all_test_losses):
        print("Losses for model with lr={0}: {1}".format(params.lrs[i], item))
        print("Accuracy for model with lr={0}: {1}".format(params.lrs[i], all_test_accs[i]))
