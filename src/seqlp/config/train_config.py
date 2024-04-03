from ..setup.tokenizer import TokenizeData

def get_config(**kwargs:dict) ->dict:
    config = {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 64,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "no_cuda": False
    }
    for key, value in kwargs.items():
        if key in config:
            config[key] = value
    return config