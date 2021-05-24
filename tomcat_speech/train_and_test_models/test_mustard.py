# test the models created in models directory with MELD data
# currently the main entry point into the system

import numpy as np

from tomcat_speech.data_prep.data_prep_helpers import (
    DatumListDataset,
    make_glove_dict,
    Glove,
)
from tomcat_speech.data_prep.mustard_data.mustard_prep import MustardPrep
from tomcat_speech.models.train_and_test_models import *
from tomcat_speech.models.plot_training import *

from tomcat_speech.models.input_models import *
from tomcat_speech.data_prep.meld_data.meld_prep import *

# import parameters for model
from tomcat_speech.models.parameters.earlyfusion_params import params

# set device
cuda = False
device = torch.device("cuda" if cuda else "cpu")

# set random seed
seed = params.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# decide if you want to use avgd feats
avgd_acoustic = params.avgd_acoustic
avgd_acoustic_in_network = params.avgd_acoustic or params.add_avging

# path to pickled dataset
# dataset = "meld_IS10_small_avgd_fullGloVe.p"

# set path to the test model
saved_model = "output/models/MUStARD_BCELoss_batch10_100hidden_2lyrs_lr0.001.pth"

if __name__ == "__main__":
    # load dataset
    glove_file = "../../glove.short.300d.punct.txt"
    # glove_file = "../../glove.42B.300d.txt"

    mustard_path = "../../datasets/multimodal_datasets/MUStARD"

    data_type = "mustard"
    # decide if you want to use avgd feats
    avgd_acoustic = params.avgd_acoustic
    avgd_acoustic_in_network = params.avgd_acoustic or params.add_avging
    # 1. IMPORT GLOVE + MAKE GLOVE OBJECT
    glove_dict = make_glove_dict(glove_file)
    glove = Glove(glove_dict)
    print("Glove object created")

    # 2. MAKE DATASET
    # meld_data = MustardPrep(mustard_path=meld_path)
    data = MustardPrep(
        mustard_path=mustard_path,
        acoustic_length=params.audio_dim,
        glove=glove,
        add_avging=params.add_avging,
        avgd=avgd_acoustic,
    )
    data.sarcasm_weights = data.sarcasm_weights.to(device)

    # get set of pretrained embeddings and their shape
    pretrained_embeddings = glove.data
    num_embeddings = pretrained_embeddings.size()[0]
    print("shape of pretrained embeddings is: {0}".format(glove.data.size()))

    # with open(dataset, "rb") as data_file:
    #     data = pickle.load(data_file)
    print("Dataset loaded")

    # get test data
    test_data = data.dev_data
    test_ds = DatumListDataset(test_data, data.sarcasm_weights)

    # create test model
    classifier = EarlyFusionMultimodalModel(
        params=params,
        num_embeddings=num_embeddings,
        pretrained_embeddings=pretrained_embeddings,
    )
    # set loss function
    loss_func = nn.BCELoss(reduction="mean")

    # get saved parameters
    classifier.load_state_dict(torch.load(saved_model))
    classifier.to(device)

    # test the model
    test_model(
        classifier,
        test_ds,
        params.batch_size,
        loss_func,
        device,
        avgd_acoustic=avgd_acoustic_in_network,
        use_speaker=params.use_speaker,
        use_gender=params.use_gender,
    )
