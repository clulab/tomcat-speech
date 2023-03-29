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

    def forward(
        self, combined_inputs, get_prob_dist=False, return_penultimate_layer=False
    ):
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


class AcousticPlusPredictionLayer(nn.Module):
    """
    Contains text processing + a prediction layer
    Needs the output of an acoustic only layer
    """

    def __init__(self, params, out_dim):
        super(AcousticPlusPredictionLayer, self).__init__()

        # specify out_dim explicity so we can do multiple tasks at once
        self.output_dim = out_dim

        self.text_input_dim = params.output_dim

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )

        self.acoustic_fc_1 = nn.Linear(params.audio_dim, 50)
        self.acoustic_fc_2 = nn.Linear(50, params.acoustic_gru_hidden_dim)

        self.fc_input_dim = self.text_input_dim + params.acoustic_gru_hidden_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, self.output_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        acoustic_len_input=None,
        get_prob_dist=False,
        return_penultimate_layer=False,
    ):
        # here, text_input is the output of the text-specific layers

        # run acoustic features through LSTM or FFNN
        if acoustic_len_input is not None:
            packed_acoustic = nn.utils.rnn.pack_padded_sequence(
                acoustic_input,
                acoustic_len_input.clamp(max=1500),
                batch_first=True,
                enforce_sorted=False,
            )
            (
                packed_acoustic_output,
                (acoustic_hidden, acoustic_cell),
            ) = self.acoustic_rnn(packed_acoustic)
            encoded_acoustic = F.dropout(acoustic_hidden[-1], self.dropout)
        else:
            if len(acoustic_input.shape) > 2:
                encoded_acoustic = acoustic_input.squeeze()
            else:
                encoded_acoustic = acoustic_input

        encoded_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_1(encoded_acoustic), self.dropout)
        )
        encoded_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_2(encoded_acoustic), self.dropout)
        )

        # combine all
        encoded_data = torch.cat((text_input, encoded_acoustic), dim=1)

        intermediate = torch.tanh(F.dropout(self.fc1(encoded_data), self.dropout))
        if return_penultimate_layer:
            penult = intermediate
        predictions = torch.relu(self.fc2(intermediate))

        if self.output_dim == 1:
            predictions = torch.sigmoid(predictions)
        elif get_prob_dist:
            prob = nn.Softmax(dim=1)
            predictions = prob(predictions)

        # return the output
        if not return_penultimate_layer:
            return predictions
        else:
            return predictions, penult


class TextPlusPredictionLayer(nn.Module):
    """
    Contains text processing + a prediction layer
    Needs the output of an acoustic only layer
    """

    def __init__(
        self,
        params,
        out_dim,
        num_embeddings=None,
        pretrained_embeddings=None,
        use_distilbert=False,
    ):
        super(TextPlusPredictionLayer, self).__init__()

        # specify out_dim explicity so we can do multiple tasks at once
        self.output_dim = out_dim

        # distilbert vs glove initialization
        self.use_distilbert = use_distilbert

        # set text input size
        if not use_distilbert:
            self.text_input_size = params.text_dim + params.short_emb_dim
        else:
            # don't use short embedding if using bert
            self.text_input_size = params.text_dim

        # if we feed text through additional layer(s)
        self.text_rnn = nn.LSTM(
            input_size=self.text_input_size,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # get number of output dims
        self.acoustic_input_dim = params.output_dim

        self.fc_input_dim = self.acoustic_input_dim + params.text_gru_hidden_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        if not use_distilbert:
            # initialize word embeddings
            self.embedding = nn.Embedding(
                num_embeddings, self.text_dim, _weight=pretrained_embeddings
            )
            self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)
            # self.text_batch_norm = nn.BatchNorm1d(self.text_dim + params.short_emb_dim)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, self.output_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        gender_input=None,
        get_prob_dist=False,
        return_penultimate_layer=False,
    ):
        # here, acoustic_input is the output of the acoustic layers

        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        if not self.use_distilbert:
            embs = F.dropout(self.embedding(text_input), 0.1).detach()
            short_embs = F.dropout(self.short_embedding(text_input), 0.1)
            all_embs = torch.cat((embs, short_embs), dim=2)
        else:
            all_embs = text_input

        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

        # combine all
        encoded_data = torch.cat((encoded_text, acoustic_input), dim=1)

        intermediate = torch.tanh(F.dropout(self.fc1(encoded_data), self.dropout))
        if return_penultimate_layer:
            penult = intermediate
        predictions = torch.relu(self.fc2(intermediate))

        if self.output_dim == 1:
            predictions = torch.sigmoid(predictions)
        elif get_prob_dist:
            prob = nn.Softmax(dim=1)
            predictions = prob(predictions)

        # return the output
        if not return_penultimate_layer:
            return predictions
        else:
            return predictions, penult
