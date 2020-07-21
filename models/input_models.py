# the models used for multimodal, multitask classification

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseGRU(nn.Module):
    """
    The basic GRU model to be used inside of the encoder
    Abstracted out since it may be used TWICE within the encoder
    (1x for text, 1x for audio)
    """
    def __init__(self, input_dim, hidden_size, output_dim, num_gru_layers,
                 num_fc_layers, dropout, bidirectional):
        super(BaseGRU, self).__init__()

        # input text
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_gru_layers = num_gru_layers
        self.dropout = dropout

        self.GRU = nn.LSTM(input_dim, output_dim, num_gru_layers, batch_first=True, dropout=dropout,
                          bidirectional=bidirectional)
        # self.GRU = nn.GRU(input_dim, output_dim, num_gru_layers, batch_first=True, dropout=dropout,
        #                   bidirectional=bidirectional)

    def forward(self, inputs, input_lengths):
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths,
                                                   batch_first=True, enforce_sorted=False)

        # todo: look at this--make sure we're taking hidden from the right place
        packed_output, (hidden, cell) = self.GRU(inputs)
        # rnn_feats, hidden = self.GRU(inputs)

        output = hidden[:,-1,:]

        # output is NOT fed through softmax or sigmoid layer here
        # assumption: output is intermediate layer of larger NN
        return output


class BasicEncoder(nn.Module):
    """
    An encoder to take a sequence of inputs and produce a sequence of intermediate representations
    Can include convolutions over text input and/or acoustic input--BUT NOT TOGETHER bc MELD isn't
    aligned at the word-level
    """
    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None):
        super(BasicEncoder, self).__init__()
        # input text + acoustic + speaker
        self.text_dim = params.text_dim
        self.audio_dim = params.audio_dim
        self.num_embeddings = num_embeddings
        self.num_speakers = params.num_speakers
        self.text_gru_hidden_dim = params.text_gru_hidden_dim

        # if we feed text through additional layer(s)
        # self.text_output_dim = params.text_output_dim
        self.text_rnn = nn.LSTM(
            input_size=params.text_dim + params.short_emb_dim,
            hidden_size=params.text_gru_hidden_dim,
            num_layers=params.num_gru_layers, 
            batch_first=True,
            bidirectional=True)

        self.acoustic_rnn = nn.LSTM(
            input_size=params.audio_dim,
            hidden_size=params.acoustic_gru_hidden_dim,
            num_layers=params.num_gru_layers,
            batch_first=True,
            bidirectional=False
        )

        # set the size of the input into the fc layers
        if params.avgd_acoustic or params.add_avging:
            self.fc_input_dim = params.text_gru_hidden_dim + params.audio_dim
        else:
            self.fc_input_dim = params.text_gru_hidden_dim + params.acoustic_gru_hidden_dim

        if params.use_speaker:
            self.fc_input_dim = self.fc_input_dim + params.speaker_emb_dim
        elif params.use_gender:
            self.fc_input_dim = self.fc_input_dim + params.gender_emb_dim

        # print(self.fc_input_dim)
        # self.fc_input_dim = params.text_output_dim

        # set number of classes
        self.output_dim = params.output_dim

        # set number of layers and dropout
        self.dropout = params.dropout

        # initialize word embeddings
        self.embedding = nn.Embedding(num_embeddings, self.text_dim,
                                      _weight=pretrained_embeddings)
        self.short_embedding = nn.Embedding(num_embeddings, params.short_emb_dim)
        # self.embedding = nn.Embedding(num_embeddings, self.text_dim)
        # self.text_batch_norm = nn.BatchNorm1d(self.text_dim + params.short_emb_dim)

        # initialize speaker embeddings
        self.speaker_embedding = nn.Embedding(params.num_speakers, params.speaker_emb_dim)

        # self.speaker_batch_norm = nn.BatchNorm1d(params.speaker_emb_dim)

        self.gender_embedding = nn.Embedding(3, params.gender_emb_dim)

        # acoustic batch normalization
        # self.acoustic_batch_norm = nn.BatchNorm1d(params.audio_dim)
        # self.acoustic_unk_norm = nn.BatchNorm1d(params.audio_dim)
        # self.acoustic_female_norm = nn.BatchNorm1d(params.audio_dim)
        # self.acoustic_male_norm = nn.BatchNorm1d(params.audio_dim)

        # initialize fully connected layers
        # self.fc1 = nn.Linear(self.fc_input_dim, params.output_dim)
        self.fc1 = nn.Linear(self.fc_input_dim, params.fc_hidden_dim)

        # self.interfc_batch_norm = nn.BatchNorm1d(params.fc_hidden_dim)

        self.fc2 = nn.Linear(params.fc_hidden_dim, params.output_dim)

    def forward(self, acoustic_input, text_input, speaker_input=None, length_input=None, acoustic_len_input=None,
                gender_input=None):
        # using pretrained embeddings, so detach to not update weights
        # embs: (batch_size, seq_len, emb_dim)
        embs = F.dropout(self.embedding(text_input), self.dropout).detach()
        # embs = self.embedding(text_input).detach()

        short_embs = F.dropout(self.short_embedding(text_input), self.dropout)
        # short_embs = self.short_embedding(text_input)

        all_embs = torch.cat((embs, short_embs), dim=2)
        # add text normalization -- must operate over dim 1 so permute
        # all_embs = self.text_batch_norm(all_embs.permute(0, 2, 1))
        # all_embs = all_embs.permute(0, 2, 1)

        # get speaker embeddings, if needed
        if speaker_input is not None:
            speaker_embs = self.speaker_embedding(speaker_input).squeeze(dim=1)
            # speaker_embs = self.speaker_batch_norm(speaker_embs)
        if gender_input is not None:
            gender_embs = self.gender_embedding(gender_input)

        # packed = nn.utils.rnn.pack_padded_sequence(embs, length_input, batch_first=True, enforce_sorted=False)
        packed = nn.utils.rnn.pack_padded_sequence(all_embs, length_input, batch_first=True, enforce_sorted=False)

        # feed embeddings through GRU
        packed_output, (hidden, cell) = self.text_rnn(packed)
        # print(hidden.shape)
        # print(packed_output.data.shape)
        # print(cell.shape)
        # sys.exit()
        # split_point = int(self.text_gru_hidden_dim / 2)
        # forward_hidden = hidden[-1, :, :split_point]
        # backward_hidden = hidden[0, :, split_point:]
        #
        # print(split_point)
        # print(forward_hidden.shape)
        # print(backward_hidden.shape)
        # sys.exit()

        # all_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        # print(all_hidden.shape)
        # sys.exit()
        # padded_output, lens = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        encoded_text = F.dropout(hidden[-1], self.dropout)
        # encoded_text = F.dropout(all_hidden, self.dropout)

        # encoded_text = hidden[-1]

        # print(encoded_text.shape)

        if acoustic_len_input is not None:
            # print(acoustic_input.shape)
            # acoustic_input = self.acoustic_batch_norm(acoustic_input.permute(0, 2, 1))
            # print(acoustic_input.shape)
            # acoustic_input = acoustic_input.permute(0, 2, 1)
            packed_acoustic = nn.utils.rnn.pack_padded_sequence(acoustic_input, acoustic_len_input, batch_first=True,
                                                                enforce_sorted=False)

            # print(packed_acoustic.data.shape)
            packed_acoustic_output, (acoustic_hidden, acoustic_cell) = self.acoustic_rnn(packed_acoustic)
            encoded_acoustic = F.dropout(acoustic_hidden[-1], self.dropout)
            # encoded_acoustic = acoustic_hidden[-1]

        else:
            # print(acoustic_input.shape)
            if len(acoustic_input.shape) > 2:
                encoded_acoustic = acoustic_input.squeeze()
            else:
                encoded_acoustic = acoustic_input
            # print(encoded_acoustic.shape)
            # encoded_acoustic = self.acoustic_batch_norm(encoded_acoustic)

        # inputs = encoded_text
        # print(encoded_acoustic.shape)
        
        # combine modalities as required by architecture
        # inputs = torch.cat((acoustic_input, encoded_text), 1)
        if speaker_input is not None:
            inputs = torch.cat((encoded_acoustic, encoded_text, speaker_embs), 1)
        elif gender_input is not None:
            inputs = torch.cat((encoded_acoustic, encoded_text, gender_embs), 1)
        else:
            inputs = torch.cat((encoded_acoustic, encoded_text), 1)

        # print(inputs.shape)
        # use pooled, squeezed feats as input into fc layers
        output = torch.tanh(F.dropout(self.fc1(inputs), self.dropout))
        # output = self.interfc_batch_norm(output)
        # todo: abstract this so it's only calculated if not multitask
        output = torch.relu(self.fc2(output))
        # output = F.softmax(output, dim=1)
        # output = torch.tanh(self.fc1(inputs))

        # return the output
        return output


class TextOnlyCNN(nn.Module):
    """
    A CNN with multiple input channels with different kernel size operating over input
    Used with only text modality.
    """
    def __init__(self, params, num_embeddings, pretrained_embeddings=None):
        super(TextOnlyCNN, self).__init__()
        # input dimensions
        self.text_dim = params.text_dim
        self.in_channels = params.text_dim

        # number of classes
        self.output_dim = params.output_dim

        # self.num_cnn_layers = params.num_cnn_layers
        self.dropout = params.dropout

        # kernels for each layer
        self.k1_size = params.kernel_1_size
        self.k2_size = params.kernel_2_size
        self.k3_size = params.kernel_3_size

        # number of output channels from conv layers
        self.out_channels = params.out_channels

        # word embeddings
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(num_embeddings, self.text_dim, padding_idx=0, max_norm=1.0)
            self.pretrained_embeddings = False
        else:
            self.embedding = nn.Embedding(num_embeddings, self.text_dim, padding_idx=0,
                                          _weight=pretrained_embeddings, max_norm=1.0)
            self.pretrained_embeddings = True

        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, self.k1_size)
        self.maxconv1 = nn.MaxPool1d(kernel_size=self.k1_size)
        self.conv2 = nn.Conv1d(self.in_channels, self.out_channels, self.k2_size)
        self.maxconv2 = nn.MaxPool1d(kernel_size=self.k2_size)
        self.conv3 = nn.Conv1d(self.in_channels, self.out_channels, self.k3_size)
        self.maxconv3 = nn.MaxPool1d(kernel_size=self.k3_size)

        # a second convolutional layer to run on top of the first ones
        # self.conv4 = nn.Conv1d(self.out_channels * 3, self.out_channels, self.k1_size)
        # self.conv5 = nn.Conv1d(self.out_channels * 3, self.out_channels, self.k2_size)
        #
        # # a third to go on the second
        # self.conv6 = nn.Conv1d(self.out_channels * 2, self.out_channels, self.k3_size)

        # fully connected layers
        self.fc1 = nn.Linear(self.out_channels * 3, params.text_cnn_hidden_dim)
        # self.fc1 = nn.Linear(self.out_channels, params.text_cnn_hidden_dim)
        self.fc2 = nn.Linear(params.text_cnn_hidden_dim, self.output_dim)

    def forward(self, acoustic_input, text_input, speaker_input=None, length_input=None,
                gender_input=None):
        # get word embeddings
        if self.pretrained_embeddings:
            # detach to avoid training them if using pretrained
            inputs = self.embedding(text_input).detach()
        else:
            inputs = self.embedding(text_input)

        inputs = inputs.permute(0, 2, 1)

        # feed data into convolutional layers
        # dim=2 says it's an unexpected argument, but it is needed for this to work
        conv1_out = F.leaky_relu(self.conv1(inputs))
        # print(conv1_out.size())
        feats1 = F.max_pool1d(conv1_out, 5, stride=1)  # .squeeze(dim=2)
        # print(feats1.shape)
        conv2_out = F.leaky_relu(self.conv2(inputs))
        feats2 = F.max_pool1d(conv2_out, 4, stride=1)  #.squeeze(dim=2)
        # print(feats1.shape)
        conv3_out = F.leaky_relu(self.conv3(inputs))
        feats3 = F.max_pool1d(conv3_out, 3, stride=1)  # .squeeze(dim=2)
        # print(feats1.shape)

        # combine output of convolutional layers
        # intermediate = torch.cat((conv1_out, conv2_out, conv3_out), 1)
        intermediate = torch.cat((feats1, feats2, feats3), 1)

        # conv4_out = F.leaky_relu(self.conv4(intermediate))
        # feats4 = F.max_pool1d(conv4_out, conv4_out.size(dim=2)).squeeze(dim=2)
        # feats4 = F.max_pool1d(conv4_out, 3)
        # conv5_out = F.leaky_relu(self.conv5(intermediate))
        # feats5 = F.max_pool1d(conv5_out, 3)

        # feats5 = F.max_pool1d(conv5_out, conv5_out.size(dim=2)).squeeze(dim=2)
        # conv6_out = F.leaky_relu(self.conv6(intermediate))
        # conv6_out = F.leaky_relu(self.conv6(torch.cat((feats4, feats5), 1)))

        #feats6
        all_feats = F.max_pool1d(intermediate, intermediate.size(dim=2)).squeeze(dim=2)

        # all_feats = torch.cat((feats4, feats5, feats6), 1)

        # feed this through fully connected layer
        fc1_out = torch.tanh(self.fc1((F.dropout(all_feats, self.dropout))))
        # fc1_out = torch.tanh(self.fc1((F.dropout(intermediate, self.dropout))))

        output = torch.relu(self.fc2(F.dropout(fc1_out, self.dropout)))
        # output = self.fc2(fc1_out)

        return output.squeeze(dim=1)


class PredictionLayer(nn.Module):
    """
    A final layer for predictions
    """
    def __init__(self, params, out_dim):
        super(PredictionLayer, self).__init__()
        self.input_dim = params.fc_hidden_dim
        self.dropout = params.dropout

        # specify out_dim explicity so we can do multiple tasks at once
        self.output_dim = out_dim

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, combined_inputs):
        out = torch.relu(self.fc1(F.dropout(combined_inputs, self.dropout)))

        return out


class MultitaskModel(nn.Module):
    """
    A model combining base + output layers for multitask learning
    """
    def __init__(self, params, num_embeddings=None, pretrained_embeddings=None,):
        super(MultitaskModel, self).__init__()
        # set base of model
        self.base = BasicEncoder(params, num_embeddings, pretrained_embeddings)

        # set output layers
        self.class_1_predictor = PredictionLayer(params, params.output_dim)
        self.class_2_predictor = PredictionLayer(params, params.output_2_dim)

    def forward(self, acoustic_input, text_input, speaker_input=None, length_input=None,
                acoustic_len_input=None, gender_input=None):
        # call forward on base model
        final_base_layer = self.base(acoustic_input, text_input, speaker_input=speaker_input,
                                     length_input=length_input, acoustic_len_input=acoustic_len_input,
                                     gender_input=gender_input)

        # get first output prediction
        class_1_out = self.class_1_predictor(final_base_layer)

        # get second output prediction
        class_2_out = self.class_2_predictor(final_base_layer)

        return class_1_out, class_2_out
