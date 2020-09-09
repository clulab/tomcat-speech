# train the models created in models directory with MUStARD data
# currently the main entry point into the system
import shutil
import sys
from datetime import date

import numpy as np

from sklearn.model_selection import train_test_split

from data_prep.ravdess_data.ravdess_prep import RavdessPrep
from models.train_and_test_models import *
from models.input_models import *
from data_prep.data_prep_helpers import *
from data_prep.meld_data.meld_prep import *
from data_prep.mustard_data.mustard_prep import *

# import parameters for model
import models.parameters.multitask_config as config
# from models.parameters.multitask_params import model_params

# set model parameters
model_params = config.model_params

sys.path.append("/net/kate/storage/work/bsharp/github/asist-speech")

# set device
cuda = False

# # Check CUDA
# if not torch.cuda.is_available():
#     cuda = False

device = torch.device("cuda" if cuda else "cpu")

# set random seed
torch.manual_seed(model_params.seed)
np.random.seed(model_params.seed)
random.seed(model_params.seed)
# if cuda:
#     torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    # decide if you want to use avgd feats
    avgd_acoustic_in_network = model_params.avgd_acoustic or model_params.add_avging

    # create save location
    output_path = os.path.join(config.exp_save_path, str(config.EXPERIMENT_ID) + "_" +
                               config.EXPERIMENT_DESCRIPTION + str(date.today()))

    # make sure the full save path exists; if not, create it
    os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(output_path))

    # copy the config file into the experiment directory
    shutil.copyfile(config.CONFIG_FILE, os.path.join(output_path, "config.py"))

    # add stdout to a log file
    with open(os.path.join(output_path, "log"), "w") as f:
        sys.stdout = f

        # 1. IMPORT GLOVE + MAKE GLOVE OBJECT
        glove_dict = make_glove_dict(config.glove_file)
        glove = Glove(glove_dict)
        print("Glove object created")

        # 2. MAKE DATASET
        mustard_data = MustardPrep(mustard_path=config.mustard_path, acoustic_length=model_params.audio_dim, glove=glove,
                                   add_avging=model_params.add_avging,
                                   use_cols=config.acoustic_columns,
                                   avgd=model_params.avgd_acoustic)

        meld_data = MeldPrep(meld_path=config.meld_path, acoustic_length=model_params.audio_dim, glove=glove,
                             add_avging=model_params.add_avging,
                             use_cols=config.acoustic_columns,
                             avgd=model_params.avgd_acoustic)

        # ravdess_data = RavdessPrep(ravdess_path=config.ravdess_path, acoustic_length=params.audio_dim, glove=glove,
        #                      add_avging=params.add_avging,
        #                      use_cols=config.acoustic_columns,
        #                      avgd=avgd_acoustic)

        # add class weights to device
        mustard_data.sarcasm_weights = mustard_data.sarcasm_weights.to(device)
        meld_data.emotion_weights = meld_data.emotion_weights.to(device)
        # ravdess_data.emotion_weights = ravdess_data.emotion_weights.to(device)

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
        for lr in model_params.lrs:
            for wd in model_params.weight_decay:

                item_output_path = os.path.join(output_path, f"LR{lr}_WD{wd}")

                # make sure the full save path exists; if not, create it
                os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(item_output_path))

                # this uses train-dev-test folds
                # create instance of model
                bimodal_trial = MultitaskModel(
                    params=model_params,
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
                mustard_multiplier = (1 - (mustard_len / total_len)) * 2
                print(f"MELD multiplier is: 1 - (meld_len / total_len) = {meld_multiplier}")
                print(f"MUStARD multiplier is: (1 - (mustard_len / total_len)) * 2 = {mustard_multiplier}")

                # add multipliers to their relevant objects
                mustard_obj.change_loss_multiplier(mustard_multiplier)
                meld_obj.change_loss_multiplier(meld_multiplier)

                # set all data list
                all_data_list = [mustard_obj, meld_obj]

                print("Model, loss function, and optimization created")

                # todo: set data sampler?
                sampler = None
                # sampler = BatchSchedulerSampler()

                # create a a save path and file for the model
                model_save_file = f"{item_output_path}/{config.EXPERIMENT_DESCRIPTION}.pth"

                # make the train state to keep track of model training/development
                train_state = make_train_state(lr, model_save_file)

                # train the model and evaluate on development set
                multitask_train_and_predict(
                    bimodal_trial,
                    train_state,
                    all_data_list,
                    model_params.batch_size,
                    model_params.num_epochs,
                    optimizer,
                    device,
                    scheduler=None,
                    sampler=sampler,
                    avgd_acoustic=avgd_acoustic_in_network,
                    use_speaker=model_params.use_speaker,
                    use_gender=model_params.use_gender,
                )

                # plot the loss and accuracy curves
                # set plot titles
                loss_title = f"Training and Dev loss for model {config.model_type} with lr {lr}"
                acc_title = f"Avg F scores for model {config.model_type} with lr {lr}"

                # set save names
                loss_save = f"{item_output_path}/loss.png"
                acc_save = f"{item_output_path}/avg_f1.png"

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

                # add best evaluation losses and accuracy from training to set
                all_test_losses.append(train_state["early_stopping_best_val"])
                # all_test_accs.append(train_state['best_val_acc'])

        # print the best model losses and accuracies for each development set in the cross-validation
        for i, item in enumerate(all_test_losses):
            print("Losses for model with lr={0}: {1}".format(model_params.lrs[i], item))
            # print("Accuracy for model with lr={0}: {1}".format(params.lrs[i], all_test_accs[i]))
