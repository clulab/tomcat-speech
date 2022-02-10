# bases to be used with the single- and multimodal models

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusionMultimodalModel(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Fuses data prior to entry into the first neural layer;
    Uses averaging of text tensors to do this
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None,
                 use_distilbert=False):
        super(EarlyFusionMultimodalModel, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.out_dims = params.output_dim

        # if we feed text through additional layer(s)
        if not use_distilbert:
            self.text_input_size = params.text_dim + params.short_emb_dim
        else:
            self.text_input_size = params.text_dim

        # set size of input dim
        self.fc_input_dim = self.text_input_size + params.audio_dim

        # initialize speaker, gender embeddings
        self.speaker_embedding = None
        self.gender_embedding = None

        if params.use_speaker:
            self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim
            self.speaker_embedding = nn.Embedding(
                params.num_speakers, params.speaker_emb_dim
            )

        elif params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim
            self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # distilbert vs glove initialization
        self.use_distilbert = use_distilbert

        if not use_distilbert:
            # initialize word embeddings
            self.embedding = nn.Embedding(
                num_embeddings, self.text_dim, _weight=pretrained_embeddings
            )
            self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        self.fc2 = nn.Linear(params.fc_hidden_dim, self.output_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
        get_prob_dist=False,
        save_encoded_data=False
    ):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        if not self.use_distilbert:
            embs = F.dropout(self.embedding(text_input), 0.1).detach()

            short_embs = F.dropout(self.short_embedding(text_input), 0.1)

            all_embs = torch.cat((embs, short_embs), dim=2)
        else:
            all_embs = text_input

        all_embs = torch.mean(all_embs, dim=1)

        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        if len(acoustic_input.shape) > 2:
            encoded_acoustic = acoustic_input.squeeze()
        else:
            encoded_acoustic = acoustic_input

        # combine modalities as required by architecture
        if speaker_input is not None:
            inputs = torch.cat((encoded_acoustic, all_embs, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_acoustic, all_embs, gender_embs), 1)
        else:
            inputs = torch.cat((encoded_acoustic, all_embs), 1)

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), self.dropout))
        output = torch.tanh(F.dropout(self.fc2(output), self.dropout))

        if self.out_dims == 1:
            output = torch.sigmoid(output)
        elif get_prob_dist:
            prob = nn.Softmax(dim=1)
            output = prob(output)

        # return the output
        return output

class IntermediateFusionMultimodalModel(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None,
                 use_distilbert=False):
        super(IntermediateFusionMultimodalModel, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.out_dims = params.output_dim

        # if we feed text through additional layer(s)
        if not use_distilbert:
            self.text_input_size = params.text_dim + params.short_emb_dim
        else:
            self.text_input_size = params.text_dim

        self.text_rnn = nn.LSTM(
            input_size=self.text_input_size,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        # self.text_batch_norm = nn.BatchNorm1d(num_features=params.text_gru_hidden_dim)

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
        )

        # set the size of the input into the fc layers
        if params.avgd_acoustic or params.add_avging:
            # set size of input dim
            self.fc_input_dim = params.text_gru_hidden_dim + params.audio_dim
            # set size of hidden
            self.fc_hidden = 50
            # set acoustic fc layer 1
            self.acoustic_fc_1 = nn.Linear(params.audio_dim, self.fc_hidden)
            # self.ac_fc_batch_norm = nn.BatchNorm1d(self.fc_hidden)
        else:
            # set size of input dim
            self.fc_input_dim = (
                params.text_gru_hidden_dim + params.acoustic_gru_hidden_dim
            )
            # set size of hidden
            self.fc_hidden = 100
            # set acoustic fc layer 1
            self.acoustic_fc_1 = nn.Linear(params.fc_hidden_dim, self.fc_hidden)

        # set acoustic fc layer 2
        self.acoustic_fc_2 = nn.Linear(self.fc_hidden, params.audio_dim)

        # initialize speaker, gender embeddings
        self.speaker_embedding = None
        self.gender_embedding = None

        if params.use_speaker:
            self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim
            self.speaker_embedding = nn.Embedding(
                params.num_speakers, params.speaker_emb_dim
            )

        elif params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim
            self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # distilbert vs glove initialization
        self.use_distilbert = use_distilbert

        if not use_distilbert:
            # initialize word embeddings
            self.embedding = nn.Embedding(
                num_embeddings, self.text_dim, _weight=pretrained_embeddings
            )
            self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)
        # self.fc_batch_norm = nn.BatchNorm1d(params.fc_hidden_dim)

        self.fc2 = nn.Linear(params.fc_hidden_dim, params.output_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
        get_prob_dist=False,
        save_encoded_data=False
    ):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        if not self.use_distilbert:
            embs = F.dropout(self.embedding(text_input), 0.1).detach()

            short_embs = F.dropout(self.short_embedding(text_input), 0.1)

            all_embs = torch.cat((embs, short_embs), dim=2)
        else:
            all_embs = text_input

        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        # flatten_parameters() decreases memory usage
        self.text_rnn.flatten_parameters()

        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)
        # encoded_text = self.text_batch_norm(encoded_text)

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
            inputs = torch.cat((encoded_acoustic, encoded_text, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_acoustic, encoded_text, gender_embs), 1)
        else:
            inputs = torch.cat((encoded_acoustic, encoded_text), 1)

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))

        if self.out_dims == 1:
            output = torch.sigmoid(output)
        elif get_prob_dist:
            prob = nn.Softmax(dim=1)
            output = prob(output)

        # return the output
        if save_encoded_data:
            return output, encoded_acoustic, encoded_text
        else:
            return output

class LateFusionMultimodalModel(nn.Module):
    """
    A late fusion model that combines modalities only at decision time
    """

    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(LateFusionMultimodalModel, self).__init__()
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim
        self.text_fc_input_dim = params.text_gru_hidden_dim + params.gender_emb_dim
        self.text_fc_hidden_dim = 100

        # get number of output dims
        self.out_dims = params.output_dim

        # if we feed text through additional layer(s)
        self.text_rnn = nn.LSTM(
            input_size=params.text_dim + params.short_emb_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # initialize fully connected layers
        self.text_fc1 = nn.Linear(self.text_fc_input_dim, self.text_fc_hidden_dim)
        self.text_fc2 = nn.Linear(self.text_fc_hidden_dim, params.output_dim)

        # initialize acoustic portions of model
        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=False,
        )

        if params.avgd_acoustic or params.add_avging:
            self.acoustic_fc_1 = nn.Linear(
                params.audio_dim + params.gender_emb_dim, 100
            )
        else:
            self.acoustic_fc_1 = nn.Linear(params.acoustic_gru_hidden_dim, 100)
        self.acoustic_fc_2 = nn.Linear(100, params.output_dim)

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize word embeddings
        self.embedding = nn.Embedding(
            num_embeddings, self.text_dim, _weight=pretrained_embeddings
        )
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(
            params.num_speakers, params.speaker_emb_dim
        )
        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        speaker_input=None,
        length_input=None,
        acoustic_len_input=None,
        gender_input=None,
    ):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        embs = F.dropout(self.embedding(text_input), 0.1).detach()
        short_embs = F.dropout(self.short_embedding(text_input), 0.1)

        all_embs = torch.cat((embs, short_embs), dim=2)

        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
            # speaker_embs = self.speaker_batch_norm(speaker_embs)
        if gender_input is not None:
            gend_embs = self.gender_embedding(gender_input)

        # packed = nn.utils.rnn.pack_padded_sequence(embs, length_input, batch_first=True, enforce_sorted=False)
        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

        if gender_input is not None:
            encoded_text = torch.cat((encoded_text, gend_embs), dim=1)

        text_intermediate = torch.tanh(
            F.dropout(self.text_fc1(encoded_text), self.dropout)
        )
        text_predictions = torch.relu(self.text_fc2(text_intermediate))

        if acoustic_len_input is not None:
            packed_acoustic = nn.utils.rnn.pack_padded_sequence(
                acoustic_input,
                acoustic_len_input,
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

        if gender_input is not None:
            encoded_acoustic = torch.cat((encoded_acoustic, gend_embs), dim=1)

        encoded_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_1(encoded_acoustic), self.dropout)
        )
        acoustic_predictions = torch.tanh(
            F.dropout(self.acoustic_fc_2(encoded_acoustic), self.dropout)
        )

        # combine predictions to get results
        # text_predictions = torch.mul(text_predictions, 4)
        # acoustic_predictions = torch.mul(acoustic_predictions, 2)
        predictions = torch.add(text_predictions, acoustic_predictions)
        # predictions = torch.mul(text_predictions, acoustic_predictions)

        if self.out_dims == 1:
            predictions = torch.sigmoid(predictions)

        # return the output
        return predictions


class MultimodalBaseDuplicateInput(nn.Module):
    """
    A multimodal model that duplicates input modality tensors
    Inspired by:
    Frustratingly easy domain adaptation
    """

    def __init__(
        self, params, num_embeddings=None, pretrained_embeddings=None, num_tasks=2
    ):
        super(MultimodalBaseDuplicateInput, self).__init__()
        # get the number of duplicates
        self.mult_size = num_tasks + 1

        # input text + acoustic + speaker
        self.text_dim = params.text_dim * self.mult_size
        self.short_dim = params.short_emb_dim * self.mult_size
        self.audio_dim = params.audio_dim * self.mult_size

        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # get number of output dims
        self.out_dims = params.output_dim

        # if we feed text through additional layer(s)
        self.text_rnn = nn.LSTM(
            input_size=self.text_dim + self.short_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        # set the size of acoustic fc layers
        self.acoustic_fc_1 = nn.Linear(self.audio_dim, 50)
        self.acoustic_fc_2 = nn.Linear(50, params.audio_dim)

        # set the size of the input into the multimodal fc layers
        self.fc_input_dim = params.text_gru_hidden_dim + params.audio_dim

        if params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize word embeddings
        self.embedding = nn.Embedding(
            num_embeddings, params.text_dim, _weight=pretrained_embeddings
        )
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(
            params.num_speakers, params.speaker_emb_dim
        )

        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # initialize fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)

    def forward(
        self,
        acoustic_input,
        text_input,
        length_input=None,
        gender_input=None,
        task_num=0,
    ):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        embs = self.embedding(text_input).detach()
        temp_embs = torch.clone(embs).detach()

        # do the same with trainable short embeddings
        short_embs = self.short_embedding(text_input)

        # add a tensor for each task; if not the tensor for this task,
        # fill it with zeros; otherwise, just copy the tensor
        for num in range(1, self.mult_size):
            # don't want to detach so recalculating short embs here
            # for safety; may not be necessary
            temp_short_embs = self.short_embedding(text_input)

            if num != task_num + 1:
                temp_embs = torch.zeros_like(temp_embs).detach()
                temp_short_embs = torch.zeros_like(temp_short_embs)

            embs = torch.cat((embs, temp_embs), dim=2)
            short_embs = torch.cat((short_embs, temp_short_embs), dim=2)

        # add dropout
        embs = F.dropout(embs, 0.1)
        short_embs = F.dropout(short_embs, 0.1)

        # concatenate short and regular embeddings
        all_embs = torch.cat((embs, short_embs), dim=2)

        # get gender embeddings, if needed
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

            for num in range(1, self.mult_size):
                temp_gend_embs = self.gender_embedding(gender_input)

                if num != task_num + 1:
                    temp_gend_embs = torch.zeros_like(temp_gend_embs)

                gender_embs = torch.cat((gender_embs, temp_gend_embs), dim=-1)

        # pack embeddings for RNN
        packed = nn.utils.rnn.pack_padded_sequence(
            all_embs, length_input, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through RNN
        packed_output, (hidden, cell) = self.text_rnn(packed)
        encoded_text = F.dropout(hidden[-1], 0.3)

        if len(acoustic_input.shape) > 2:
            acoustic_input = acoustic_input.squeeze()

        temp_acoustic_input = torch.clone(acoustic_input).detach()

        for num in range(1, self.mult_size):
            # i THINK detach here just keeps differentiation separate from the original
            # could check both ways to verify
            if num != task_num + 1:
                temp_acoustic_input = torch.zeros_like(temp_acoustic_input)

            acoustic_input = torch.cat((acoustic_input, temp_acoustic_input), dim=1)

        int_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_1(acoustic_input), self.dropout)
        )
        encoded_acoustic = torch.tanh(
            F.dropout(self.acoustic_fc_2(int_acoustic), self.dropout)
        )

        # combine modalities as required by architecture
        if gender_input is not None:
            inputs = torch.cat((encoded_acoustic, encoded_text, gender_embs), 1)
        else:
            inputs = torch.cat((encoded_acoustic, encoded_text), 1)

        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), 0.5))

        if self.out_dims == 1:
            output = torch.sigmoid(output)

        # return the output
        return output
