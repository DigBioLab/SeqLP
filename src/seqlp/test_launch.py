from src.seqlp.setup.train_model import TrainModel
from src.seqlp.setup.tokenizer import TokenizeData


####INPUTS#####
command_script_dir = r"C:\Users\nilsh\my_projects\SeqLP\data"
store_dir = r"C:\Users\nilsh\my_projects\SeqLP\NLP_train"
###############
tokenize = TokenizeData()

filename = tokenize.download_and_prepare(download_commands_script=command_script_dir,
                                    limit = 10000,
                                    save_single_csvs = False,
                                    user_dir = store_dir)
train_encodings, val_encodings = tokenize.tokenize(filename)

heavy_config = {
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

params = {
"num_train_epochs": 2,
"per_device_train_batch_size": 16,
"per_device_eval_batch_size": 64,
"warmup_steps": 500,
"weight_decay": 0.01,
"no_cuda": True
}
Train = TrainModel(train_encodings,
                     val_encodings,
                     tokenize.data_collator,
                   model_config=heavy_config,
                   train_params=params,
                   user_dir=store_dir)
Train.train()