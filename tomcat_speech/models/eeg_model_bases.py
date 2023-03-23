import torch
import torch.nn as nn
import torch.nn.functional as F


class EEG_Model_Base(nn.Module):
    def __init__(self, params):
        super(EEG_Model_Base, self).__init__()

        self.num_channels = params.num_eeg_channels
        self.num_waves = params.num_eeg_waves

        # todo: incorporate possibility for tcn
        # can use generic LSTM to start with
        # should be bidirectional
        self.lstm = nn.LSTM(
            input_size=self.num_channels * self.num_waves,  # todo: this might not be correct
            hidden_size=params.eeg_lstm_hidden_dim,
            num_layers=params.num_eeg_lstm_layers,
            batch_first=True,
            bidirectional=True)

        # set acoustic fc layer 1
        self.fc_1 = nn.Linear(params.eeg_lstm_hidden_dim, params.eeg_fc_hidden_dim)
        self.fc_2 = nn.Linear(params.eeg_fc_hidden_dim, params.eeg_out_dim)

    def forward(self, eeg_batch, eeg_lengths):
        # flatten_parameters() decreases memory usage
        self.lstm.flatten_parameters()

        packed = nn.utils.rnn.pack_padded_sequence(
            eeg_batch, eeg_lengths, batch_first=True, enforce_sorted=False
        )

        # feed embeddings through GRU
        packed_output, (encoded, cell) = self.lstm(packed)

        # use pooled, squeezed feats as input into fc layers
        intermediate = torch.tanh(F.dropout(self.fc1(encoded), 0.3))
        out = F.dropout(self.fc2(encoded), 0.3)

        return out


