from .setup.train_model import TrainModel
heavy_config = {
    "num_hidden_layers": 3,
    "num_attention_heads": 3,
    "hidden_size": 768,
    "d_ff": 3072,
    "vocab_size": 25,
    "max_len": 150,
    "max_position_embeddings": 152,
    "batch_size": 96,
    "max_steps": 225000,
    "weight_decay": 0.01,
    "peak_learning_rate": 0.0001,
    }

params = {
"num_train_epochs": 2,
"per_device_train_batch_size": 16,
"per_device_eval_batch_size": 64,
"warmup_steps": 500,
"weight_decay": 0.01,
"no_cuda": True
}
Train = TrainModel(limit_files=2, download_commands_script=r"C:\Users\nilsh\my_projects\SeqLP\data", model_config=heavy_config, train_params=params)
