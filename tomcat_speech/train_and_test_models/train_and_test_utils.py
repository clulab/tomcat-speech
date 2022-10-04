
import torch
import numpy as np
import random

from tomcat_speech.models.multimodal_models import MultitaskModel, MultitaskAcousticShared, \
    MultitaskDuplicateInputModel, MultitaskTextShared
from tomcat_speech.models.text_model_bases import TextOnlyModel

def set_cuda_and_seeds(config):
    # set cuda
    cuda = False
    if torch.cuda.is_available():
        cuda = True

    device = torch.device("cuda" if cuda else "cpu")

    # # Check CUDA
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # set random seed
    torch.manual_seed(config.model_params.seed)
    np.random.seed(config.model_params.seed)
    random.seed(config.model_params.seed)
    if cuda:
        torch.cuda.manual_seed_all(config.model_params.seed)

    # check if cuda
    print(cuda)

    if cuda:
        # check which GPU used
        print(torch.cuda.current_device())

    return device


def select_model(model_params, num_embeddings, pretrained_embeddings, multidataset=True):
    """
    Use model parameters to select the appropriate model
    Return this model for training
    """
    # set embeddings to None if using bert -- they are calculated
    #   anyway, so if you don't do this, it will ALWAYS use embeddings
    if model_params.use_distilbert:
        num_embeddings = None
        pretrained_embeddings = None

    if "acoustic_shared" in model_params.model.lower():
        model = MultitaskAcousticShared(params=model_params,
                                        use_distilbert=model_params.use_distilbert,
                                        num_embeddings=num_embeddings,
                                        pretrained_embeddings=pretrained_embeddings)
    elif "text_shared" in  model_params.model.lower():
        model = MultitaskTextShared(params=model_params,
                                    use_distilbert=model_params.use_distilbert,
                                    num_embeddings=num_embeddings,
                                    pretrained_embeddings=pretrained_embeddings)
    elif "text_only" in model_params.model.lower():
        model = TextOnlyModel(params=model_params,
                use_distilbert=model_params.use_distilbert,
                num_embeddings=num_embeddings,
                pretrained_embeddings=pretrained_embeddings)
    elif "duplicate_input" in model_params.model.lower():
        model = MultitaskDuplicateInputModel(params=model_params,
                                             use_distilbert=model_params.use_distilbert,
                                             num_embeddings=num_embeddings,
                                             pretrained_embeddings=pretrained_embeddings)
    else:
        model = MultitaskModel(params=model_params,
                               use_distilbert=model_params.use_distilbert,
                               num_embeddings=num_embeddings,
                               pretrained_embeddings=pretrained_embeddings,
                               multi_dataset=multidataset)

     return model
