parameter = dict(
    model_cfg = dict(
        nhead = 3,
        hidden_dim = 128,
        num_layers = 3,
        hidden_neuron = 1,
        max_len = 2200,
         pretrained_from = '/transformer_checkpoint/model_class_bound[14]_data[13,15]_h3_l3_balance_checkpoint.pth'
    ),
    
    dataset_cfg = dict(
         boundary = 14,
         path = '/data/tokens.json',
    ),
    
    train_cfg = dict(
        base_lr = 1e-4,
        epoch = 72,
        milestone = [50],
        gamma = 0.1,
        batch = 12
    ),
    
    valid_cfg = dict(
        batch = 2
    )
)
