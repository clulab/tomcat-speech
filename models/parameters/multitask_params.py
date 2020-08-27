from argparse import Namespace

params = Namespace(
    # consistency parameters
    seed=88,  # 1007
    # trying text only model or not
    text_only=False,
    # overall model parameters
    model="Multitask-mustard",
    num_epochs=20,
    batch_size=10,  # 128,  # 32
    early_stopping_criteria=20,
    num_gru_layers=2,  # 1,   # 3,  # 1,  # 4, 2,
    bidirectional=False,
    # input dimension parameters
    text_dim=300,  # text vector length
    short_emb_dim=30,  # length of trainable embeddings vec
    audio_dim=10,  # 76,  # 79,  # 10 # audio vector length
    # audio_dim=10,
    # text NN
    # text_output_dim=30,   # 100,   # 50, 300,
    text_gru_hidden_dim=100,  # 30,  # 50,  # 20
    # text-only CNN
    kernel_1_size=3,
    kernel_2_size=4,
    kernel_3_size=5,
    out_channels=512,
    text_cnn_hidden_dim=100,
    # acoustic NN
    avgd_acoustic=False,  # set true to use avgd acoustic feat vectors without RNN
    add_avging=True,  # set to true if you want to avg acoustic feature vecs upon input
    acoustic_gru_hidden_dim=100,
    # speaker embeddings
    use_speaker=False,
    num_speakers=261,  # check this number
    speaker_emb_dim=3,
    # gender embeddings
    use_gender=True,
    gender_emb_dim=4,
    # outputs
    output_dim=20,  # output dimensions from last layer of base model
    output_0_dim=1,
    output_1_dim=7,  # 7,  # length of output vector
    output_2_dim=0,  # 3,    # length of second task output vec
    output_3_dim=0,
    # FC layer parameters
    num_fc_layers=1,  # 1,  # 2,
    fc_hidden_dim=100,  # 20,
    dropout=0.4,  # 0.2
    # optimizer parameters
    lrs=[1e-4],
    beta_1=0.9,
    beta_2=0.999,
    weight_decay=[0.0001],
)
