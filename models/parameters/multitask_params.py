from argparse import Namespace

params = Namespace(
            # consistency parameters
            seed=888,  # 1007

            # overall model parameters
            model="Multitask-meld",
            num_epochs=100,
            batch_size=45,  # 32
            early_stopping_criteria=10,

            num_gru_layers=4,  # 4, 2,
            bidirectional=False,

            # input dimension parameters
            text_dim=300,  # text vector length
            audio_dim=76,  # 79,  # 10 # audio vector length

            # text NN
            text_output_dim=25,   # 100,   # 50, 300,
            text_gru_hidden_dim=15,  # 50,  # 20

            # outputs
            output_dim=7,  # length of output vector

            # FC layer parameters
            num_fc_layers=1,  # 1,  # 2,
            fc_hidden_dim=15,
            dropout=0.5,  # 0.2

            # optimizer parameters
            lrs=[1e-4],
            beta_1=0.9,
            beta_2=0.999,  
            weight_decay=0.01,
)
