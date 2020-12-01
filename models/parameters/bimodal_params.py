from argparse import Namespace

params = Namespace(
    # consistency parameters
    seed=888,  # 1007
    cols_to_skip=3,
    # alignment None with internal data
    #   and "utt" with external asist data
    alignment=None,
    # overall model parameters
    model="BimodalCNN",
    num_splits=4,  # 5  #splits for CV
    num_epochs=300,
    batch_size=4,
    early_stopping_criteria=1500,
    # input dimension parameters
    text_dim=300,  # text vector length
    # audio_dim=512, # 10 # audio_train vector length
    audio_dim=[1,512,1522],
    num_gru_layers=2,
    acoustic_gru_hidden_dim=1024,
    bidirectional=True,
    # speaker parameters
    spkr_emb_dim=1,
    gender_emb_dim=2,
    num_speakers=2,
    use_speaker=False,
    use_gender=False,
    # number of classes
    output_dim=2,  # length of output vector
    # CNN-specific parameters
    num_layers=3,  # 3  # number of lstm/cnn layers
    out_channels=20,
    kernel_size=3,
    padding_idx=0,
    # FC layer parameters
    num_fc_layers=1,
    fc_hidden_dim=128,
    dropout=0.0,  # 0.2
    # optimizer parameters
    # learning_rate=1e-06,  # 1e-06 0.00001  # 0.0001 0.001 tried
    lrs=[1e-06],  # if using multiple learning rates to test
    beta_1=0.9,
    beta_2=0.999,  # beta params for Adam--defaults 0.9 and 0.999
    weight_decay=[0.0001],
    # scheduler parameters (if using)
    use_scheduler=False,
    scheduler_factor=0.5,
    scheduler_patience=3,
)
