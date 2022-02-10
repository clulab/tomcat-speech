
import torch
import torch.nn as nn
import torch.nn.functional as F


class AcousticOnlyModel(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    """

    def __init__(self, params):
        super(AcousticOnlyModel, self).__init__()
        # input acoustic + speaker
        self.audio_dim = params.audio_dim
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.out_dims = params.output_dim

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )

        # set the size of the input into the fc layers
        self.fc_input_dim = params.audio_dim

        if params.add_avging is False and params.avgd_acoustic is False:
            self.acoustic_fc_1 = nn.Linear(params.fc_hidden_dim, 100)
        else:
            self.acoustic_fc_1 = nn.Linear(params.audio_dim, 50)
        self.acoustic_fc_2 = nn.Linear(50, params.audio_dim)

        if params.use_speaker:
            self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim
        elif params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(
            params.num_speakers, params.speaker_emb_dim
        )

        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, params.output_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
    ):
        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        # if adding noise
        # noise = torch.ones(acoustic_input.shape)
        # noise = noise.normal_()
        # noise = torch.normal(0, 0.2, size=acoustic_input.shape[1])
        # acoustic_input = torch.add(acoustic_input, noise[:, None])

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

        # combine modalities as required by architecture
        if speaker_input is not None:
            inputs = torch.cat((encoded_acoustic, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_acoustic, gender_embs), 1)
        else:
            inputs = encoded_acoustic

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))

        if self.out_dims == 1:
            output = torch.sigmoid(output)

        # return the output
        return output
