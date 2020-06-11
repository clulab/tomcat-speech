from argparse import Namespace

params = Namespace(
            # consistency parameters
            seed=888,  # 1007
            cols_to_skip=3,

            # overall model parameters
            model="Multitask",
            num_splits=4,  # 5  #splits for CV
            num_epochs=100,
            batch_size=10,
            early_stopping_criteria=50,
            text_network=True,
            alignment="utt",

            # input dimension parameters
            text_dim=300,  # text vector length
            audio_dim=79,  # 10 # audio vector length

            # speaker parameters
            spkr_emb_dim=1,
            num_speakers=260,  # num speakers per split specified in meld readme, but not overall num
            use_speaker=True,

            # text CNN
            num_text_conv_layers=1,
            num_text_fc_layers=1,
            text_out_channels=100,
            text_output_dim=300,

            # audio CNN
            num_audio_conv_layers=1,
            num_audio_fc_layers=1,
            audio_out_channels=20,
            audio_output_dim=10,

            # outputs
            output_dim=7,  # length of output vector
            softmax=True,

            # CNN-specific parameters
            num_layers=3,  # 3  # number of lstm/cnn layers
            out_channels=20,
            kernel_size=3,
            stride=1,
            padding_idx=0,

            # FC layer parameters
            num_fc_layers=1,
            hidden_dim=50,
            dropout=0.0,  # 0.2

            # optimizer parameters
            # learning_rate=1e-06,  # 1e-06 0.00001  # 0.0001 0.001 tried
            lrs=[1e-05],  # if using multiple learning rates to test
            beta_1=0.9,
            beta_2=0.999,  # beta params for Adam--defaults 0.9 and 0.999
            weight_decay=0.01,

            # scheduler parameters (if using)
            use_scheduler=False,
            scheduler_factor=0.5,
            scheduler_patience=3
)