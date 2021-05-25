# train the models created in models directory with MUStARD data
# currently the main entry point into the system
import shutil
import sys
from datetime import date
import pickle

import numpy as np
import copy
from sklearn.model_selection import train_test_split

from tomcat_speech.models.chalearn_models import OCEANPersonalityModel

# from sklearn.model_selection import train_test_split

from tomcat_speech.data_prep.chalearn_data.chalearn_prep import ChalearnPrep
from tomcat_speech.models.train_and_test_models import *
from tomcat_speech.models.plot_training import *
from tomcat_speech.data_prep.data_prep_helpers import *
from tomcat_speech.data_prep.meld_data.meld_prep import *

# import parameters for model
import tomcat_speech.models.parameters.chalearn_config as config

# from models.parameters.multitask_params import model_params

# set device
cuda = False

if torch.cuda.is_available():
    cuda = True

device = torch.device("cuda" if cuda else "cpu")

# # Check CUDA
if torch.cuda.is_available():
    torch.cuda.set_device(2)

# set random seed

torch.manual_seed(config.model_params.seed)
np.random.seed(config.model_params.seed)
random.seed(config.model_params.seed)
if cuda:
    torch.cuda.manual_seed_all(config.model_params.seed)

if __name__ == "__main__":
    # check if cuda
    print(cuda)

    if cuda:
        # check which GPU used
        print(torch.cuda.current_device())

    # decide if you want to use avgd feats
    avgd_acoustic_in_network = (
        config.model_params.avgd_acoustic or config.model_params.add_avging
    )

    # create save location
    output_path = os.path.join(
        config.exp_save_path,
        str(config.EXPERIMENT_ID)
        + "_"
        + config.EXPERIMENT_DESCRIPTION
        + str(date.today()),
    )

    # set location for pickled data (saving or loading)
    if config.USE_SERVER:
        data = "/data/nlp/corpora/MM/pickled_data"
    else:
        data = "data"

    # make sure the full save path exists; if not, create it
    os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(output_path))

    # copy the config file into the experiment directory
    shutil.copyfile(config.CONFIG_FILE, os.path.join(output_path, "config.py"))

    # add stdout to a log file
    with open(os.path.join(output_path, "log"), "w") as f:
        # todo: make this flush more frequently so you can check the bottom of the log file
        #   or make a new function e.g. print_both and have it both print and save to file
        sys.stdout = f

        if not config.load_dataset:
            # 0. CHECK TO MAKE SURE DATA DIRECTORY EXISTS
            os.system(f'if [ ! -d "{data}" ]; then mkdir -p {data}; fi')

            # 1. IMPORT GLOVE + MAKE GLOVE OBJECT
            glove_dict = make_glove_dict(config.glove_file)
            glove = Glove(glove_dict)
            print("Glove object created")

            # 2. MAKE DATASET
            chalearn_data = ChalearnPrep(
                chalearn_path=config.chalearn_path,
                acoustic_length=config.model_params.audio_dim,
                glove=glove,
                add_avging=config.model_params.add_avging,
                use_cols=config.acoustic_columns,
                avgd=config.model_params.avgd_acoustic,
                f_end=f"_{config.feature_set}.csv",
                pred_type=config.chalearn_predtype,
                # utts_file_name="chalearn_kaldi.tsv"
            )

            # add class weights to device
            if config.chalearn_predtype == "max_class":
                chalearn_data.trait_weights = chalearn_data.trait_weights.to(device)
            elif (
                config.chalearn_predtype == "high-low"
                or config.chalearn_predtype == "high-med-low"
                or config.chalearn_predtype == "binary"
                or config.chalearn_predtype == "ternary"
            ):
                chalearn_data.neur_weights = chalearn_data.neur_weights.to(device)
                chalearn_data.openn_weights = chalearn_data.openn_weights.to(device)
                chalearn_data.extr_weights = chalearn_data.extr_weights.to(device)
                chalearn_data.agree_weights = chalearn_data.agree_weights.to(device)
                chalearn_data.consc_weights = chalearn_data.consc_weights.to(device)
                chalearn_data.trait_weights = None

            # get train, dev, test partitions
            chalearn_train_ds = DatumListDataset(
                chalearn_data.train_data, "chalearn_traits", chalearn_data.trait_weights
            )
            chalearn_dev_ds = DatumListDataset(
                chalearn_data.dev_data, "chalearn_traits", chalearn_data.trait_weights
            )
            chalearn_test_ds = DatumListDataset(
                chalearn_data.test_data, "chalearn_traits", chalearn_data.trait_weights
            )

            chalearn_train_ds_5000, _ = train_test_split(
                chalearn_train_ds, train_size=5000
            )
            chalearn_train_ds_4000, _ = train_test_split(
                chalearn_train_ds_5000, train_size=4000
            )
            chalearn_train_ds_3000, _ = train_test_split(
                chalearn_train_ds_4000, train_size=3000
            )
            chalearn_train_ds_2000, _ = train_test_split(
                chalearn_train_ds_3000, train_size=2000
            )
            chalearn_train_ds_1000, _ = train_test_split(
                chalearn_train_ds_2000, train_size=1000
            )
            chalearn_train_ds_900, _ = train_test_split(
                chalearn_train_ds_1000, train_size=900
            )
            chalearn_train_ds_800, _ = train_test_split(
                chalearn_train_ds_900, train_size=800
            )
            chalearn_train_ds_700, _ = train_test_split(
                chalearn_train_ds_900, train_size=700
            )
            chalearn_train_ds_600, _ = train_test_split(
                chalearn_train_ds_900, train_size=600
            )
            chalearn_train_ds_500, _ = train_test_split(
                chalearn_train_ds_900, train_size=500
            )
            chalearn_train_ds_400, _ = train_test_split(
                chalearn_train_ds_900, train_size=400
            )
            chalearn_train_ds_300, _ = train_test_split(
                chalearn_train_ds_900, train_size=300
            )
            chalearn_train_ds_200, _ = train_test_split(
                chalearn_train_ds_900, train_size=200
            )
            chalearn_train_ds_100, _ = train_test_split(
                chalearn_train_ds_900, train_size=100
            )
            chalearn_train_ds_50, _ = train_test_split(
                chalearn_train_ds_900, train_size=50
            )
            chalearn_train_ds_10, _ = train_test_split(
                chalearn_train_ds_50, train_size=10
            )
            chalearn_train_ds_5, _ = train_test_split(
                chalearn_train_ds_10, train_size=5
            )
            chalearn_train_ds_1, _ = train_test_split(chalearn_train_ds_5, train_size=1)
            print("CHALEARN data split")

            if config.save_dataset:
                # save all data for faster loading
                save_path = data + "/" + config.load_path

                # make sure the full save path exists; if not, create it
                os.system('if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(save_path))

                # save all data for faster loading
                pickle.dump(
                    chalearn_train_ds_5000,
                    open(f"{save_path}/chalearn_IS1013_5000_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_4000,
                    open(f"{save_path}/chalearn_IS1013_4000_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_3000,
                    open(f"{save_path}/chalearn_IS1013_3000_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_2000,
                    open(f"{save_path}/chalearn_IS1013_2000_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_1000,
                    open(f"{save_path}/chalearn_IS1013_1000_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_900,
                    open(f"{save_path}/chalearn_IS1013_900_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_800,
                    open(f"{save_path}/chalearn_IS1013_800_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_700,
                    open(f"{save_path}/chalearn_IS1013_700_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_600,
                    open(f"{save_path}/chalearn_IS1013_600_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_500,
                    open(f"{save_path}/chalearn_IS1013_500_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_400,
                    open(f"{save_path}/chalearn_IS1013_400_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_300,
                    open(f"{save_path}/chalearn_IS1013_300_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_200,
                    open(f"{save_path}/chalearn_IS1013_200_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_100,
                    open(f"{save_path}/chalearn_IS1013_100_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_50,
                    open(f"{save_path}/chalearn_IS1013_50_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_10,
                    open(f"{save_path}/chalearn_IS1013_10_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_5,
                    open(f"{save_path}/chalearn_IS1013_5_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds_1,
                    open(f"{save_path}/chalearn_IS1013_1_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_train_ds,
                    open(f"{save_path}/chalearn_IS1013_train.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_dev_ds,
                    open(f"{save_path}/chalearn_IS1013_dev.pickle", "wb"),
                )
                pickle.dump(
                    chalearn_test_ds,
                    open(f"{save_path}/chalearn_IS1013_test.pickle", "wb"),
                )

                # pickle.dump(
                #     glove, open("data/glove.pickle", "wb")
                # )
                exit()

            print("Datasets created")

        else:
            # 1. Load datasets + glove object
            chalearn_train_ds = pickle.load(
                open(f"{data}/{config.load_path}/chalearn_IS1013_1_train.pickle", "rb")
            )
            chalearn_dev_ds = pickle.load(
                open(f"{data}/{config.load_path}/chalearn_IS1013_dev.pickle", "rb")
            )
            chalearn_test_ds = pickle.load(
                open(f"{data}/{config.load_path}/chalearn_IS1013_test.pickle", "rb")
            )

            print("ChaLearn data loaded")

            # 1. IMPORT GLOVE + MAKE GLOVE OBJECT
            glove_dict = make_glove_dict(config.glove_file)
            glove = Glove(glove_dict)
            print("Glove object created")

            # # load glove
            # glove = pickle.load(open("data/glove.pickle", "rb"))

            print("GloVe object loaded")

        # 3. CREATE NN
        # get set of pretrained embeddings and their shape
        pretrained_embeddings = glove.data
        num_embeddings = pretrained_embeddings.size()[0]
        print("shape of pretrained embeddings is: {0}".format(glove.data.size()))

        # prepare holders for loss and accuracy of best model versions
        all_test_losses = []
        all_test_accs = []

        # mini search through different learning_rate values etc.
        for lr in config.model_params.lrs:
            for b_size in config.model_params.batch_size:
                for num_gru_layer in config.model_params.num_gru_layers:
                    for short_emb_size in config.model_params.short_emb_dim:
                        for output_d in config.model_params.output_dim:
                            for dout in config.model_params.dropout:
                                for (
                                    txt_hidden_dim
                                ) in config.model_params.text_gru_hidden_dim:

                                    this_model_params = copy.deepcopy(
                                        config.model_params
                                    )

                                    this_model_params.batch_size = b_size
                                    this_model_params.num_gru_layers = num_gru_layer
                                    this_model_params.short_emb_dim = short_emb_size
                                    this_model_params.output_dim = output_d
                                    this_model_params.dropout = dout
                                    this_model_params.text_gru_hidden_dim = (
                                        txt_hidden_dim
                                    )

                                    print(this_model_params)

                                    item_output_path = os.path.join(
                                        output_path,
                                        f"LR{lr}_BATCH{b_size}_"
                                        f"NUMLYR{num_gru_layer}_"
                                        f"SHORTEMB{short_emb_size}_"
                                        f"INT-OUTPUT{output_d}_"
                                        f"DROPOUT{dout}_"
                                        f"TEXTHIDDEN{txt_hidden_dim}",
                                    )

                                    # make sure the full save path exists; if not, create it
                                    os.system(
                                        'if [ ! -d "{0}" ]; then mkdir -p {0}; fi'.format(
                                            item_output_path
                                        )
                                    )

                                    # this uses train-dev-test folds
                                    # create instance of model
                                    multitask_model = OCEANPersonalityModel(
                                        params=this_model_params,
                                        num_embeddings=num_embeddings,
                                        pretrained_embeddings=pretrained_embeddings,
                                    )

                                    optimizer = torch.optim.Adam(
                                        lr=lr,
                                        params=multitask_model.parameters(),
                                        weight_decay=this_model_params.weight_decay,
                                    )

                                    # set the classifier(s) to the right device
                                    multitask_model = multitask_model.to(device)
                                    print(multitask_model)

                                    # add loss function for chalearn
                                    chalearn_loss_func = nn.CrossEntropyLoss(
                                        # weight=chalearn_train_ds.class_weights,
                                        reduction="mean"
                                    )
                                    # create multitask object
                                    chalearn_obj = MultitaskObject(
                                        chalearn_train_ds,
                                        chalearn_dev_ds,
                                        chalearn_test_ds,
                                        chalearn_loss_func,
                                        task_num=0,
                                    )

                                    print(
                                        "Model, loss function, and optimization created"
                                    )

                                    # todo: set data sampler?
                                    sampler = None
                                    # sampler = BatchSchedulerSampler()

                                    # create a a save path and file for the model
                                    model_save_file = f"{item_output_path}/{config.EXPERIMENT_DESCRIPTION}.pth"

                                    # make the train state to keep track of model training/development
                                    train_state = make_train_state(lr, model_save_file)

                                    # train the model and evaluate on development set
                                    if config.chalearn_predtype == "max_class":
                                        max_class = True
                                    else:
                                        max_class = False
                                    #
                                    # print(max_class)
                                    # print(config.chalearn_predtype)
                                    # sys.exit()

                                    personality_as_multitask_train_and_predict(
                                        multitask_model,
                                        train_state,
                                        chalearn_train_ds,
                                        chalearn_dev_ds,
                                        this_model_params.batch_size,
                                        this_model_params.num_epochs,
                                        chalearn_loss_func,
                                        optimizer,
                                        device,
                                        scheduler=None,
                                        sampler=sampler,
                                        avgd_acoustic=avgd_acoustic_in_network,
                                        use_speaker=this_model_params.use_speaker,
                                        use_gender=this_model_params.use_gender,
                                        max_class=max_class,
                                    )
                                    # plot the loss and accuracy curves
                                    # set plot titles
                                    loss_title = f"Training and Dev loss for model {config.model_type} with lr {lr}"
                                    loss_save = f"{item_output_path}/loss.png"

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

                                    # plot the avg f1 curves for each dataset
                                    for item in train_state["tasks"]:
                                        plot_train_dev_curve(
                                            train_state["train_avg_f1"][item],
                                            train_state["val_avg_f1"][item],
                                            x_label="Epoch",
                                            y_label="Weighted AVG F1",
                                            title=f"Average f-scores for task {item} for model {config.model_type} with lr {lr}",
                                            save_name=f"{item_output_path}/avg-f1_task-{item}.png",
                                            losses=False,
                                            set_axis_boundaries=False,
                                        )
