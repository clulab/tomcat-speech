
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionLayer(nn.Module):
    """
    A final layer for predictions
    """

    def __init__(self, params, out_dim):
        super(PredictionLayer, self).__init__()
        self.input_dim = params.output_dim
        self.inter_fc_prediction_dim = params.final_hidden_dim
        self.dropout = params.dropout

        # specify out_dim explicity so we can do multiple tasks at once
        self.output_dim = out_dim

        # self.fc1 = nn.Linear(self.input_dim, self.output_dim)
        self.fc1 = nn.Linear(self.input_dim, self.inter_fc_prediction_dim)
        self.fc2 = nn.Linear(self.inter_fc_prediction_dim, self.output_dim)

    def forward(self, combined_inputs, get_prob_dist=False, return_penultimate_layer=False):
        out = torch.relu(F.dropout(self.fc1(combined_inputs), self.dropout))
        if return_penultimate_layer:
            penult = out
        out = torch.relu(self.fc2(out))
        # out = torch.relu(self.fc1(F.dropout(combined_inputs, self.dropout)))

        if self.output_dim == 1:
            out = torch.sigmoid(out)
        elif get_prob_dist:
            prob = nn.Softmax(dim=1)
            out = prob(out)

        if not return_penultimate_layer:
            return out
        else:
            return out, penult
