from ..setup.tokenizer import TokenizeData

def get_config(tokenize:TokenizeData, **kwargs:dict) ->dict:
    config = {
        "num_hidden_layers": 3,
        "num_attention_heads": 3,
        "hidden_size": 768,
        "d_ff": 3072,
        "vocab_size": len(tokenize.tokenizer), # important for input layer
        "max_len": 150,
        "max_position_embeddings": 152,
        "batch_size": 96,
        "max_steps": 225000,
        "weight_decay": 0.01,
        "peak_learning_rate": 0.0001,
        }
    for key, value in kwargs.items():
        if key in config:
            config[key] = value
    return config