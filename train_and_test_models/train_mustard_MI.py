# train the models created in models directory with MELD data
# currently the main entry point into the system

import sys

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.append("/work/seongjinpark/tomcat-speech")

from models.train_and_test_models import *

from models.input_models_sjp import *
from data_prep.data_prep_helpers import *
from data_prep.mustard_data.mustard_w2v_prep import *

from models.parameters.bimodal_params import params

# import parameters for model

# set device
cuda = True

# # Check CUDA
# if not torch.cuda.is_available():
#     cuda = False

device = torch.device("cuda:1" if cuda else "cpu")

# set random seed
seed = params.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# if cuda:
#     torch.cuda.manual_seed_all(seed)

# set parameters for data prep
# todo: should be updated later to a glove subset appropriate for this task

print(os.getcwd())
# mustard_path = "/data/nlp/corpora/MM/MELD_five_dialogues"
# mustard_path = "/data/nlp/corpora/MM/MELD_formatted"
mustard_path = "/data/seongjinpark/MUStARD/"
# meld_path = "data"
mustard_data_path = "/work/seongjinpark/tomcat-speech/data/mustard"
# wav_model = "/work/seongjinpark/tomcat-speech/data/wav2vec_large.pt"
# meld_path = "../../datasets/multimodal_datasets/MELD_five_utterances"
# meld_path = "../../datasets/multimodal_datasets/MUStARD"

data_type = "mustard"
# set model name and model type
model = params.model
# path to directory where best models are saved
model_save_path = "/work/seongjinpark/tomcat-speech/output/models/"
# model_save_path = "output/models/"
# make sure the full save path exists; if not, create it
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_save_path))
# set dir to plot the loss/accuracy curves for training
# model_plot_path = "/work/seongjinpark/tomcat-speech/output/plots/"
model_plot_path = "output/plots/"
os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(model_plot_path))

if __name__ == "__main__":

    # 1. MAKE DATASET

    data = MustardPrep(
        mustard_path=mustard_path,
        mustard_data_path=mustard_data_path,
        rnn=False
    )

    # add class weights to device
    train_data = data.get_train()
    dev_data = data.get_dev()

    print("Dataset created")

    # prepare holders for loss and accuracy of best model versions
    all_test_losses = []

    # mini search through different learning_rate values
    for lr in params.lrs:
        for wd in params.weight_decay:
            # model_type = f"Multitask_1.6vs1lossWeighting_Adagrad_TextOnly_100batch_wd{str(wd)}_.2split"
            # model_type = f"TextOnly_smallerPool_100batch_wd{str(wd)}_.2split_500hidden"
            # model_type = f"AcousticGenderAvgd_noBatchNorm_.2splitTrainDev_IS10avgdAI_100batch_wd{str(wd)}_30each"
            # model_type = "DELETE_ME_extraAudioFCs_.4drpt_Acou20Hid100Out"
            model_type = (
                "MUSTARD-W2V-BD-ATTN-LSTM"
            )

            # this uses train-dev-test folds
            multi_model = MultiAcousticModel(params=params)

            optimizer = torch.optim.Adam(
                lr=lr, params=multi_model.parameters(), weight_decay=wd
            )

            # set the classifier(s) to the right device
            multi_model = multi_model.to(device)
            print(multi_model)

            # set loss function, optimization, and scheduler, if using
            loss_func = nn.CrossEntropyLoss(reduction="mean")
            # loss_func = nn.BCELoss(reduction="mean")
            # loss_func = nn.CrossEntropyLoss(data.emotion_weights, reduction='mean')
            # optimizer = torch.optim.SGD(bimodal_trial.parameters(), lr=lr, momentum=0.9)

            print("Model, loss function, and optimization created")

            # set the train, dev, and set data
            # train_data = data.train_data

            # create a a save path and file for the model
            model_save_file = "{0}_batch{1}_{2}hidden_2lyrs_lr{3}.pth".format(
                model_type, params.batch_size, params.fc_hidden_dim, lr
            )

            # make the train state to keep track of model training/development
            train_state = make_train_state(lr, os.path.join(model_save_path, model_save_file))

            # train the model and evaluate on development set
            train_and_predict_multi(
                multi_model,
                train_state,
                train_data,
                dev_data,
                params.batch_size,
                params.num_epochs,
                loss_func,
                optimizer,
                device,
                scheduler=None,
                sampler=None,
                binary=False
            )

            # plot the loss and accuracy curves
            # set plot titles
            loss_title = "Training and Dev loss for model {0} with lr {1}".format(
                model_type, lr
            )
            acc_title = "Avg F scores for model {0} with lr {1}".format(model_type, lr)

            # set save names
            loss_save = "/work/seongjinpark/tomcat-speech/output/plots/{0}_lr{1}_loss.png".format(model_type, lr)
            acc_save = "/work/seongjinpark/tomcat-speech/output/plots/{0}_lr{1}_avg_f1.png".format(model_type, lr)

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
