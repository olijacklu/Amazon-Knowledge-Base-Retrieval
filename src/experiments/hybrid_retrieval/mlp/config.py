PARAM_GRID = {
    'hidden_dims': [
        [512, 256],
        [512, 128],
        [256, 128],
        [256, 64],
        [128, 64],
        [64, 32],
        [512, 256, 128],
        [256, 128, 64],
        [128, 64, 32]
    ],
    'learning_rate': [0.001, 0.0001],
    'batch_size': [32, 64, 128],
    'dropout': [0.2, 0.3, 0.4, 0.5],
    'weight_decay': [0.0001, 0.001, 0.01],
    'use_batch_norm': [True, False]
}

INPUT_DIM = 1024
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
