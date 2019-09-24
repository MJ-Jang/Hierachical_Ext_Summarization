class ModelConfig:
    # path
    data_path = ''
    tok_path = './tokenizer.model' # sentencepiece model
    model_path = './resource/HiTransformer.pkl'

    # params
    max_sent_len = 128
    max_doc_len = 64
    ignore_index_ext = 2

    d_model = 256
    emb_dim = 100
    heads = 8
    N = 2
    
    alpha = 0.85
    gamma = 2

    # training
    cuda = True
    lr = 1e-6
    batch_size = 128
    n_epoch = 30
