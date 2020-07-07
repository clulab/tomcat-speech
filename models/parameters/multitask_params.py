from argparse import Namespace

params = Namespace(
            # consistency parameters
            seed=88,  # 1007

            # overall model parameters
            model="Multitask-meld",
            num_epochs=100,
            batch_size=128,  # 32
            early_stopping_criteria=30,

            num_gru_layers=1,  # 4, 2,
            bidirectional=False,

            # input dimension parameters
            text_dim=300,  # text vector length
            short_emb_dim=30,  # length of trainable embeddings vec
            audio_dim=76,  # 79,  # 10 # audio vector length

            # text NN
            # text_output_dim=30,   # 100,   # 50, 300,
            text_gru_hidden_dim=30,  # 50,  # 20

            # acoustic NN
            acoustic_gru_hidden_dim=20,

            # outputs
            output_dim=7,  # length of output vector

            # FC layer parameters
            num_fc_layers=1,  # 1,  # 2,
            fc_hidden_dim=20,
            dropout=0.5,  # 0.2

            # optimizer parameters
            lrs=[1e-2],
            beta_1=0.9,
            beta_2=0.999,  
            weight_decay=0.00,
)
