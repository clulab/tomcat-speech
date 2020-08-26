# test the models created in models directory with MELD data
# currently the main entry point into the system

import pickle
import numpy as np

from data_prep.data_prep_helpers import DatumListDataset
from models.train_and_test_models import *

from models.input_models import *
from data_prep.data_prep import *
from data_prep.meld_data.meld_prep import *

# import parameters for model
from models.parameters.earlyfusion_params import params

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
dataset = "meld_IS10_small_avgd_fullGloVe.p"

# set path to the test model
saved_model = "output/models/TextOnly_FullGloVe_100batch_wd0.0001_.2split_batch100_100hidden_2lyrs_lr0.001.pth"

if __name__ == "__main__":

    # load dataset
    with open(dataset, "rb") as data_file:
        data = pickle.load(data_file)
    print("Dataset loaded")

    # get test data
    test_data = data.test_data
    test_ds = DatumListDataset(test_data, data.emotion_weights)

    # get text embedding info
    pretrained_embeddings = data.glove.data
    num_embeddings = pretrained_embeddings.size()[0]

    # create test model
    classifier = TextOnlyCNN(
        params=params,
        num_embeddings=num_embeddings,
        pretrained_embeddings=pretrained_embeddings,
    )

    # get saved parameters
    classifier.load_state_dict(torch.load(saved_model))
    classifier.to(device)

    # set loss function
    loss_func = nn.CrossEntropyLoss(reduction="mean")

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
