# train the models created in models directory with MUStARD data
# currently the main entry point into the system

import sys
import numpy as np

from sklearn.model_selection import train_test_split

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

mustard_path = "../../datasets/multimodal_datasets/MUStARD"
meld_path = "../../datasets/multimodal_datasets/MELD_formatted"
# meld_path = "../../datasets/multimodal_datasets/MELD_five_dialogues"

data_type = "multitask"
fusion_type = "early"

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
    mustard_data = MustardPrep(mustard_path=mustard_path, acoustic_length=params.audio_dim, glove=glove,
                               add_avging=params.add_avging,
                    use_cols=['pcm_loudness_sma', 'F0finEnv_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
                              'shimmerLocal_sma', 'pcm_loudness_sma_de', 'F0finEnv_sma_de',
                              'voicingFinalUnclipped_sma_de', 'jitterLocal_sma_de', 'shimmerLocal_sma_de'],
                    avgd=avgd_acoustic)

    meld_data = MeldPrep(meld_path=meld_path, acoustic_length=params.audio_dim, glove=glove,
                         add_avging=params.add_avging,
                         use_cols=['pcm_loudness_sma', 'F0finEnv_sma', 'voicingFinalUnclipped_sma', 'jitterLocal_sma',
                                   'shimmerLocal_sma', 'pcm_loudness_sma_de', 'F0finEnv_sma_de',
                                   'voicingFinalUnclipped_sma_de', 'jitterLocal_sma_de', 'shimmerLocal_sma_de'],
                         avgd=avgd_acoustic)

    # add class weights to device
    mustard_data.sarcasm_weights = mustard_data.sarcasm_weights.to(device)
    meld_data.emotion_weights = meld_data.emotion_weights.to(device)

    print("Datasets created")

    # 3. CREATE NN
    # get set of pretrained embeddings and their shape
    pretrained_embeddings = glove.data
    num_embeddings = pretrained_embeddings.size()[0]
    print("shape of pretrained embeddings is: {0}".format(glove.data.size()))

    # prepare holders for loss and accuracy of best model versions
    all_test_losses = []
    all_test_accs = []

    # mini search through different learning_rate values
    for lr in params.lrs:
        for wd in params.weight_decay:
            model_type = "MULTITASK_TEST"

            # this uses train-dev-test folds
            # create instance of model
            bimodal_trial = MultitaskModel(
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

            # combine train and dev data
            # ready data for mustard
            mustard_train_ds = DatumListDataset(mustard_data.train_data, "mustard", mustard_data.sarcasm_weights)
            mustard_dev_ds = DatumListDataset(mustard_data.dev_data, "mustard", mustard_data.sarcasm_weights)
            mustard_test_ds = DatumListDataset(mustard_data.test_data, "mustard", mustard_data.sarcasm_weights)

            # add loss function for mustard
            mustard_loss_func = nn.BCELoss(reduction="mean")

            # create multitask object
            mustard_obj = MultitaskObject(mustard_train_ds, mustard_dev_ds, mustard_test_ds, mustard_loss_func,
                                          task_num=0, binary=True)

            meld_train_ds = DatumListDataset(meld_data.train_data, "meld_emotion", meld_data.emotion_weights)
            meld_dev_ds = DatumListDataset(meld_data.dev_data, "meld_emotion", meld_data.emotion_weights)
            meld_test_ds = DatumListDataset(meld_data.test_data, "meld_emotion", meld_data.emotion_weights)

            meld_loss_func = nn.CrossEntropyLoss(reduction="mean")

            meld_obj = MultitaskObject(meld_train_ds, meld_dev_ds, meld_test_ds, meld_loss_func, task_num=1)

            # calculate lengths of train sets and use this to determine multipliers for the loss functions
            mustard_len = len(mustard_train_ds)
            meld_len = len(meld_train_ds)

            total_len = mustard_len + meld_len
            meld_multiplier = 1 - (meld_len / total_len)
            mustard_multiplier = 1 - (mustard_len / total_len)

            # add multipliers to their relevant objects
            mustard_obj.change_loss_multiplier(mustard_multiplier)
            meld_obj.change_loss_multiplier(meld_multiplier)


            # set all data list
            all_data_list = [mustard_obj, meld_obj]

            print("Model, loss function, and optimization created")

            # todo: set data sampler
            sampler = None
            # sampler = BatchSchedulerSampler()

            # create a a save path and file for the model
            # fixme: update to be more comprehensive
            model_save_file = "{0}_batch{1}_{2}hidden_2lyrs_lr{3}.pth".format(
                model_type, params.batch_size, params.fc_hidden_dim, lr
            )

            # make the train state to keep track of model training/development
            # todo: need a separate train state for each dataset?
            train_state = make_train_state(lr, model_save_path, model_save_file)
            # mustard_train_state = make_train_state(lr, model_save_path, model_save_file)
            # meld_train_state = make_train_state(lr, model_save_path, model_save_file)

            # train the model and evaluate on development set
            multitask_train_and_predict(
                bimodal_trial,
                train_state,
                all_data_list,
                params.batch_size,
                params.num_epochs,
                optimizer,
                device,
                scheduler=None,
                sampler=sampler,
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
