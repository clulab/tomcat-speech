
import sys

import torch
import torch.nn as nn
from tomcat_speech.models.audio_model_bases import AcousticOnlyForMultitask
from tomcat_speech.models.model_prediction_layers import PredictionLayer


class MultitaskAcousticOnly(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """

    def __init__(self, params, multi_dataset=True):
        super(MultitaskAcousticOnly, self).__init__()
        # save whether there are multiple datasets
        # if so, assumes each dataset has its own task
        self.multi_dataset = multi_dataset

        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.acoustic_base = AcousticOnlyForMultitask(
            params,
            multi_dataset,
            use_rnn=(not (params.avgd_acoustic or params.add_avging)),
        )

        # set output layers
        self.task_0_predictor = PredictionLayer(params, params.output_0_dim)
        self.task_1_predictor = PredictionLayer(params, params.output_1_dim)
        self.task_2_predictor = PredictionLayer(params, params.output_2_dim)
        self.task_3_predictor = PredictionLayer(params, params.output_3_dim)
        self.task_4_predictor = PredictionLayer(params, params.output_4_dim)

    def forward(
        self,
        acoustic_input,
        acoustic_len_input=None,
        task_num=0,
        get_prob_dist=False,
        return_penultimate_layer=False,
        save_encoded_data=False
    ):
        # call forward on base model
        final_base_layer = self.acoustic_base(
            acoustic_input, length_input=acoustic_len_input,
        )

        # set task-specific outputs
        task_0_out = None
        task_1_out = None
        task_2_out = None
        task_3_out = None
        task_4_out = None

        # get predictions for each task
        if not self.multi_dataset:
            task_0_out = self.task_0_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_1_out = self.task_1_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_2_out = self.task_2_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_3_out = self.task_3_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            task_4_out = self.task_4_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
        else:
            if task_num == 0:
                task_0_out = self.task_0_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 1:
                task_1_out = self.task_1_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 2:
                task_2_out = self.task_2_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 3:
                task_3_out = self.task_3_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            elif task_num == 4:
                task_4_out = self.task_4_predictor(final_base_layer, get_prob_dist, return_penultimate_layer)
            else:
                sys.exit(f"Task {task_num} not defined")

        if save_encoded_data:
            return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out, final_base_layer
        else:
            return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out
