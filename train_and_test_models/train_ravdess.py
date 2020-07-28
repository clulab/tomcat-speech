# train the network on ravdess input

import sys
import numpy as np
from sklearn.model_selection import train_test_split

from data_prep.ravdess_data.ravdess_prep import RavdessPrep
from models.train_and_test_models import *

from models.input_models import *
from data_prep.data_prep_helpers import *
from data_prep.meld_data.meld_prep import *
from data_prep.mustard_data.mustard_prep import *

# import parameters for model
from models.parameters.multitask_params import params

sys.path.append("/net/kate/storage/work/bsharp/github/asist-speech")

# set device
cuda = False

# # Check CUDA
# if not torch.cuda.is_available():
#     cuda = False

device = torch.device("cuda" if cuda else "cpu")

# set random seed
seed = params.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# if cuda:
#     torch.cuda.manual_seed_all(seed)

# set parameters for data prep
# todo: should be updated later to a glove subset appropriate for this task
# glove_file = "/work/bsharp/glove.short.300d.punct.txt"
# glove_file = "/data/nlp/corpora/glove/glove.840B.300d.no_proc_header.txt"
glove_file = "../../glove.short.300d.punct.txt"
# glove_file = "../../glove.42B.300d.txt"

ravdess_path = "../../datasets/multimodal_datasets/RAVDESS_Speech/IS10"

data_type = "ravdess_emotion"

# set model name and model type
model = params.model
# path to directory where best models are saved
model_save_path = "output/models/"
# make sure the full save path exists; if not, create it
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_save_path))
# set dir to plot the loss/accuracy curves for training
model_plot_path = "output/plots/"
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_plot_path))

# decide if you want to use avgd feats
avgd_acoustic = params.avgd_acoustic
avgd_acoustic_in_network = params.avgd_acoustic or params.add_avging


if __name__ == "__main__":

    # 1. IMPORT GLOVE + MAKE GLOVE OBJECT
    glove_dict = make_glove_dict(glove_file)
    glove = Glove(glove_dict)
    print("Glove object created")

    # 2. MAKE DATASET
    data = RavdessPrep(ravdess_path=ravdess_path, acoustic_length=params.audio_dim, glove=glove,
                       add_avging=params.add_avging,
                       use_cols=['pcm_loudness_sma', 'F0finEnv_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
                              'shimmerLocal_sma', 'pcm_loudness_sma_de', 'F0finEnv_sma_de',
                              'voicingFinalUnclipped_sma_de', 'jitterLocal_sma_de', 'shimmerLocal_sma_de'],
                       avgd=avgd_acoustic)

    # add class weights to device
    data.emotion_weights = data.emotion_weights.to(device)
    data.intensity_weights = data.intensity_weights.to(device)

    print("Dataset created")

    # 3. CREATE NN
    # get set of pretrained embeddings and their shape
    pretrained_embeddings = glove.data
    num_embeddings = pretrained_embeddings.size()[0]
    print("shape of pretrained embeddings is: {0}".format(glove.data.size()))

    # prepare holders for loss and accuracy of best model versions
    all_test_losses = []

    # mini search through different learning_rate values
    for lr in params.lrs:
        for wd in params.weight_decay:
            model_type = "RAVDESS_IS10sm_500txthid_.1InDrpt_.3textdrpt_.4acdrpt_.5finalFCdrpt"

            # this uses train-dev-test folds
            # create instance of model
            # TODO: WE WILL WANT TO REPLACE THIS WITH AN ACOUSTIC-ONLY MODEL, PROBABLY
            bimodal_trial = BasicEncoder(
                params=params,
                num_embeddings=num_embeddings,
                pretrained_embeddings=pretrained_embeddings,
            )
            optimizer = torch.optim.Adam(
                lr=lr, params=bimodal_trial.parameters(), weight_decay=wd
            )

            # set the classifier(s) to the right device
            bimodal_trial = bimodal_trial.to(device)
            print(bimodal_trial)

            # set loss function, optimization, and scheduler, if using
            loss_func = nn.CrossEntropyLoss(reduction="mean")
            # loss_func = nn.CrossEntropyLoss(data.emotion_weights, reduction='mean')

            print("Model, loss function, and optimization created")

            # set the train, dev, and set data
            # train_data = data.train_data

            # combine train and dev data
            # combine train and dev data
            train_ds = DatumListDataset(data.train_data, data_type, data.emotion_weights)
            dev_ds = DatumListDataset(data.dev_data, data_type, data.emotion_weights)
            test_ds = DatumListDataset(data.test_data, data_type, data.emotion_weights)

            #
            # train_ds = data.train_data
            # dev_ds = data.dev_data
            # test_ds = data.test_data

            # create a a save path and file for the model
            model_save_file = "{0}_batch{1}_{2}hidden_2lyrs_lr{3}.pth".format(
                model_type, params.batch_size, params.fc_hidden_dim, lr
            )

            # make the train state to keep track of model training/development
            train_state = make_train_state(lr, model_save_path, model_save_file)

            # train the model and evaluate on development set
            train_and_predict(
                bimodal_trial,
                train_state,
                train_ds,
                dev_ds,
                params.batch_size,
                params.num_epochs,
                loss_func,
                optimizer,
                device,
                scheduler=None,
                sampler=None,
                avgd_acoustic=avgd_acoustic_in_network,
                use_speaker=params.use_speaker,
                use_gender=params.use_gender,
            )

            # plot the loss and accuracy curves
            # set plot titles
            loss_title = "Training and Dev loss for model {0} with lr {1}".format(
                model_type, lr
            )
            acc_title = "Avg F scores for model {0} with lr {1}".format(model_type, lr)

            # set save names
            loss_save = "output/plots/{0}_lr{1}_loss.png".format(model_type, lr)
            acc_save = "output/plots/{0}_lr{1}_avg_f1.png".format(model_type, lr)

            # plot the loss from model
            plot_train_dev_curve(
                train_state["train_loss"],
                train_state["val_loss"],
                x_label="Epoch",
                y_label="Loss",
                title=loss_title,
                save_name=loss_save,
                set_axis_boundaries=False,
            )
            # plot the accuracy from model
            plot_train_dev_curve(
                train_state["train_avg_f1"],
                train_state["val_avg_f1"],
                x_label="Epoch",
                y_label="Weighted AVG F1",
                title=acc_title,
                save_name=acc_save,
                losses=False,
                set_axis_boundaries=False,
            )

            # plot_train_dev_curve(train_state['train_acc'], train_state['val_acc'], x_label="Epoch",
            #                         y_label="Accuracy", title=acc_title, save_name=acc_save, losses=False,
            #                         set_axis_boundaries=False)

            # add best evaluation losses and accuracy from training to set
            all_test_losses.append(train_state["early_stopping_best_val"])
            # all_test_accs.append(train_state['best_val_acc'])

    # print the best model losses and accuracies for each development set in the cross-validation
    for i, item in enumerate(all_test_losses):
        print("Losses for model with lr={0}: {1}".format(params.lrs[i], item))
        # print("Accuracy for model with lr={0}: {1}".format(params.lrs[i], all_test_accs[i]))
