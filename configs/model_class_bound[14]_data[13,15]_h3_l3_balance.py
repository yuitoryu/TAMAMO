parameter = dict(
    model_cfg = dict(
        nhead = 3,
        hidden_dim = 128,
        num_layers = 3,
        hidden_neuron = 1,
        max_len = 2200,
        pretrained_from = None
    ),
    
    dataset_cfg = dict(
         boundary = 14,
         path = '/data/tokens.json',
    ),
    
    train_cfg = dict(
        base_lr = 1e-3,
        epoch = 72,
        milestone = [23,60],
        gamma = 0.1,
        batch = 12
    ),
    
    valid_cfg = dict(
        batch = 2
    )
)
