# models to be used specifically with the personality trait data from chalearn

import torch.nn as nn
from tomcat_speech.models.input_models import EarlyFusionMultimodalModel, PredictionLayer


# class RegressionLayer(nn.Module):
#     """
#     A final layer for regression predictions
#     """
#
#     def __init__(self, params, out_dim):
#         super(RegressionLayer, self).__init__()
#         self.input_dim = params.output_dim
#         self.inter_fc_prediction_dim = params.final_hidden_dim
#         self.dropout = params.dropout
#
#         # specify out_dim explicity so we can do multiple tasks at once
#         self.output_dim = out_dim
#
#         self.fc1 = nn.Linear(self.input_dim, self.inter_fc_prediction_dim)
#         self.fc2 = nn.Linear(self.inter_fc_prediction_dim, 1)
#
#     def forward(self, combined_inputs):
#         out = torch.relu(F.dropout(self.fc1(combined_inputs), self.dropout))
#         out = torch.relu(self.fc2(out))
#         # out = torch.relu(self.fc1(F.dropout(combined_inputs, self.dropout)))
#
#         # because
#         if self.output_dim == 1:
#             out = torch.sigmoid(out)
#
#         return out


class OCEANPersonalityModel(nn.Module):
    """
    A model combining base + output layers for learning OCEAN personality traits
    Assumes a different prediction made for each personality trait
    Formulated as a multitask problem (each trait = 1 task)
    DO NOT use this with max-class identification or regression
    todo: add regression option? would only impact prediction layer
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(OCEANPersonalityModel, self).__init__()

        # # set base of model
        # comment this out and uncomment the below to try late fusion model
        self.base = EarlyFusionMultimodalModel(
            params, num_embeddings, pretrained_embeddings
        )

        # uncomment this and comment the above to try the late fusion model
        # self.base = LateFusionMultimodalModel(
        #     params, num_embeddings, pretrained_embeddings
        # )

        # set output layers
        # each output is the length of the prediction vector
        self.task_0_predictor = PredictionLayer(params, params.final_output_dim)
        self.task_1_predictor = PredictionLayer(params, params.final_output_dim)
        self.task_2_predictor = PredictionLayer(params, params.final_output_dim)
        self.task_3_predictor = PredictionLayer(params, params.final_output_dim)
        self.task_4_predictor = PredictionLayer(params, params.final_output_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None
    ):
        # call forward on base model
        final_base_layer = self.base(
            acoustic_input,
            text_input,
            speaker_input=speaker_input,
            length_input=length_input,
            acoustic_len_input=acoustic_len_input,
            gender_input=gender_input,
        )

        task_0_out = self.task_0_predictor(final_base_layer)
        task_1_out = self.task_1_predictor(final_base_layer)
        task_2_out = self.task_2_predictor(final_base_layer)
        task_3_out = self.task_3_predictor(final_base_layer)
        task_4_out = self.task_4_predictor(final_base_layer)

        return task_0_out, task_1_out, task_2_out, task_3_out, task_4_out
