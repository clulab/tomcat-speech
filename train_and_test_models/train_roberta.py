# test the models created in models directory with MELD data
# currently the main entry point into the system

import sys

sys.path.append("/work/seongjinpark/tomcat-speech/")

import os
import torch
from transformers import AdamW
from fairseq.models.roberta import RobertaModel

import numpy as np

from data_prep.roberta_data_prep import AudioOnlyData
from models.train_and_test_models import *

from models.input_models import *
from data_prep import *
# from data_prep.meld_data.meld_prep import *

# import parameters for model
from models.parameters.earlyfusion_params import params

# set device
cuda = True
device = torch.device("cuda:0" if cuda else "cpu")

# set random seed
seed = params.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# set path to the pre-trained models
vq_wav2vec = "/work/seongjinpark/tomcat-speech/data/vq-wav2vec_kmeans.pt"
roberta_kmeans = "/work/seongjinpark/tomcat-speech/data/bert_kmeans.pt"

# set path to the test model
saved_model = "/work/seongjinpark/tomcat-speech/output/models/AudioOnly_Roberta.pth"

# prepare holders for loss and accuracy of best model versions
all_test_losses = []

if __name__ == "__main__":

    model_type = "ROBERTa"

    # load dataset
    if os.path.exists("/work/seongjinpark/tomcat-speech/data/data_pt/train.pt"):
        train_data = torch.load("/work/seongjinpark/tomcat-speech/data/data_pt/train.pt")
    else:
        train_dataset = AudioOnlyData(audio_path="/work/seongjinpark/tomcat-speech/data/audio_train",
                                      audio_token_path="/work/seongjinpark/tomcat-speech/data/audio_train_token",
                                      response_data="/work/seongjinpark/tomcat-speech/data/train_sent_emo.csv")
        train_data = torch.utils.data.DataLoader(train_dataset, num_workers=0, batch_size=4, shuffle=True)
        
        with open("/work/seongjinpark/tomcat-speech/data/data_pt/train.pt", "wb") as data_file:
            torch.save(train_data, data_file)

    if os.path.exists("/work/seongjinpark/tomcat-speech/data/data_pt/dev.pt"):
        dev_data = torch.load("/work/seongjinpark/tomcat-speech/data/data_pt/dev.pt")
    else:
        dev_dataset = AudioOnlyData(audio_path="/work/seongjinpark/tomcat-speech/data/audio_dev",
				    audio_token_path="/work/seongjinpark/tomcat-speech/data/audio_dev_token",
                                    response_data="/work/seongjinpark/tomcat-speech/data/dev_sent_emo.csv")
        dev_data = torch.utils.data.DataLoader(dev_dataset, num_workers=0, batch_size=4, shuffle=True)
        
        with open("/work/seongjinpark/tomcat-speech/data/data_pt/dev.pt", "wb") as data_file:
            torch.save(dev_data, data_file)

    print("Dataset loaded")

    # create Model
    model = RobertaModel.from_pretrained("/work/seongjinpark/tomcat-speech/data/", "bert_kmeans.pt")
    model.register_classification_head('meld_emotion', num_classes=7)
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    print(model)
    print(model.parameters)

    lr = 0.00001
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

    # model_save_path = "models/outputs"
    model_save_file = "/work/seongjinpark/tomcat-speech/models/outputs/roberta.pth"

    train_state = make_train_state(lr, model_save_file)

    train_and_predict_transformer(classifier=model,
                                  train_state=train_state,
                                  train_ds=train_data,
                                  val_ds=dev_data,
                                  # batch_size=16,
                                  # num_workers=0,
                                  num_epochs=3,
                                  optimizer=optimizer,
                                  loss_func=loss_func,
                                  device=device)


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
    # all_test_losses.append(train_state["early_stopping_best_val"])
    # all_test_accs.append(train_state['best_val_acc'])

    # for i, item in enumerate(all_test_losses):
    #     print("Losses for model with lr={0}: {1}".format(params.lrs[i], item))
        # print("Accuracy for model with lr={0}: {1}".format(params.lrs[i], all_test_accs[i]))
