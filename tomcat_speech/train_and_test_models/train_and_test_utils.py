
import torch
import numpy as np
import random


def set_cuda_and_seeds(config):
    # set cuda
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

    # check if cuda
    print(cuda)

    if cuda:
        # check which GPU used
        print(torch.cuda.current_device())

    return device
