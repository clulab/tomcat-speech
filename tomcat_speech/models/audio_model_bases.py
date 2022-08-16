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
            speaker_input=None,
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


class SpecCNNBase(nn.Module):
    """
    A CNN base to use with spectrogram data; DOES NOT contain any linear layers
    """

    def __init__(self, params):
        super(SpecCNNBase, self).__init__()

        self.spec_dim = 513  # todo: check this number

        self.output_dim = params.spec_out_dim

        # kernels for each layer
        self.k1_size = params.kernel_1_size
        self.k2_size = params.kernel_2_size
        self.k3_size = params.kernel_3_size

        # number of output channels from conv layers
        self.out_channels = params.out_channels

        #self.conv1 = nn.Conv1d(self.spec_dim, 256, self.k1_size)
        #self.maxconv1 = nn.MaxPool1d(kernel_size=self.k1_size)
        #self.conv2 = nn.Conv1d(256, self.out_channels * 3, self.k2_size)
        #self.maxconv2 = nn.MaxPool1d(kernel_size=self.k2_size)
        #self.conv3 = nn.Conv1d(self.out_channels * 3, self.out_channels, self.k3_size)
        #self.maxconv3 = nn.MaxPool1d(kernel_size=self.k3_size)

        # this was taken from a CNN over features, but is now over a spectrogram
        # so we actually need a different shape for the input, i think
        self.conv1 = nn.Conv2d(1, 6, self.k1_size)
        self.maxconv1 = nn.MaxPool2d(kernel_size=self.k1_size)
        self.conv2 = nn.Conv2d(6, 16, self.k2_size)
        self.maxconv2 = nn.MaxPool2d(kernel_size=self.k2_size)
        self.conv3 = nn.Conv2d(16, 32, self.k3_size)
        self.maxconv3 = nn.MaxPool2d(kernel_size=self.k3_size)

        self.fc1 = nn.Linear(32 * 55 * 32, self.out_channels)  # 54 * 30 is resulting size of matrix after conv3

        def forward(self, spec_input):
            inputs = spec_input.permute(0, 2, 1)
            inputs = inputs.unsqueeze(dim=1)  # testing this
            # print(f"shape of inputs after unsqueeze is {inputs.shape}")
            # feed data into convolutional layers
            # conv1_out = F.leaky_relu(self.conv1(inputs))
            # feats1 = F.max_pool1d(conv1_out, 5, stride=1)

            # conv2_out = F.leaky_relu(self.conv2(inputs))
            # feats2 = F.max_pool1d(conv2_out, 4, stride=1)

            # conv3_out = F.leaky_relu(self.conv3(inputs))
            # feats3 = F.max_pool1d(conv3_out, 3, stride=1)

            conv1_out = F.leaky_relu(self.conv1(inputs))
            feats1 = self.maxconv1(conv1_out)
            # feats1 = F.max_pool2d(conv1_out, 5) # , stride=1)
            # print(f"shape of conv1_out is { conv1_out.shape}")
            feats1 = self.maxconv1(conv1_out)
            # feats1 = F.max_pool2d(conv1_out, 5) # , stride=1)
            # print(f"shape of conv1_out is { conv1_out.shape}")

            conv2_out = F.leaky_relu(self.conv2(conv1_out))
            feats2 = self.maxconv2(conv2_out)
            # feats2 = F.max_pool2d(conv2_out, 3) #, stride=1)
            # print(f"shape of feats2 is {feats2.shape}")
            conv3_out = F.leaky_relu(self.conv3(feats2))
            feats3 = self.maxconv3(conv3_out)
            # feats3 = F.max_pool2d(conv3_out, 3) #, stride=1)
            # feats3 = F.max_pool2d(conv3_out, kernel_size = (conv3_out.shape[2], int(round(conv3_out.shape[3]/3))))
            # print(f"shape of feats3 is {feats3.shape}")
            # combine output of convolutional layers

            # intermediate = torch.cat((feats1, feats2, feats3), 1)
            # all_feats = feats3.squeeze(dim=3)
            all_feats = feats3.flatten(start_dim=1)
            # print(f"shape of feats just before fc1 is {all_feats.shape}")
            # feats6

            # intermediate = feats3.squeeze(dim=2)
            # all_feats = F.max_pool1d(intermediate, intermediate.size(dim=2)).squeeze(dim=2)

            # add fc layer for testing purposes
            all_feats = self.fc1(all_feats)

            return all_feats


class SpecOnlyCNN(nn.Module):
    """
    A CNN with multiple input channels with different kernel size operating over input
    Used with only spectrogram modality.
    """

    def __init__(self, params):
        super(SpecOnlyCNN, self).__init__()

        # get base cnn
        self.cnn = SpecCNNBase(params)

        # fully connected layers
        self.fc1 = nn.Linear(self.out_channels * 3, params.text_cnn_hidden_dim)
        self.fc2 = nn.Linear(params.text_cnn_hidden_dim, self.output_dim)

    def forward(
            self,
            spec_input
    ):
        all_feats = self.cnn(spec_input)

        # feed this through fully connected layer
        fc1_out = torch.tanh(self.fc1((F.dropout(all_feats, self.dropout))))

        output = self.fc2(fc1_out)

        return output.squeeze(dim=1)
