from transformers import Trainer, TrainingArguments
from .setup_model import SetupModel
import os
from transformers import BertTokenizer
from . import Prepare
import gzip
import shutil
import pandas as pd

class TrainModel:
    heavy_config = {
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
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
    "num_train_epochs": 5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 64,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "no_cuda": True
    }
    def __init__(self, download_commands_script, model_config = heavy_config,train_params = params, limit_files = 10, user_dir = None) -> None:
        self.user_dir = user_dir
        gz_filename = self.download_and_prepare(download_commands_script, limit = limit_files)
        if os.path.isfile(gz_filename):
            self.train_encodings, self.val_encodings = self.tokenize(gz_filename)
            self.model = self.model_setup(heavy_config = model_config)
            self.train_params = self.train_args(train_params)
            self.trainer = self.setup_trainer()
        else:
            print("File not found. Please check the path to the file.")
    
    @staticmethod
    def setup_dirs(user_dir):
        if user_dir == None:
            dir = os.getcwd()
        else:
            dir = user_dir
        result_dir = os.path.join(dir, "results")
        if os.path.isdir(result_dir):
            pass
        else:
            os.mkdir(result_dir)
        log_dir = os.path.join(dir, "logs")
        if os.path.isdir(log_dir):
            pass
        else:
            os.mkdir(log_dir)
        return result_dir, log_dir

    @staticmethod
    def download_and_prepare(download_commands_script:str, limit = 1000000, save_single_csvs = False) -> str:            
        Prep = Prepare(download_commands_script)
        concatenated_df = Prep.download_data(limit = limit, save_single_csvs = save_single_csvs)
        concatenated_df = Prep.create_uniform_series(concatenated_df)
        concatenated_df = Prep.drop_duplicates(concatenated_df)
        filename = os.path.join(Prep.save_dir, "concatenated.csv")
        concatenated_df.to_csv(filename, index = False)
        with open(filename, 'rb') as f_in:
            with gzip.open(filename + '.gz', 'wb') as f_out:
                  shutil.copyfileobj(f_in, f_out)
        os.remove(filename)
        return filename + '.gz'
    
    @staticmethod
    def read_gzipped_csv(filename):
        with gzip.open(filename, 'rt') as f:
            df = pd.read_csv(f)
        return df
    
    def tokenize(self, gz_filename:str):
        concatenated_df = self.read_gzipped_csv(gz_filename)
        train_sequences, val_sequences = Prepare.create_train_test(concatenated_df)
        tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case = False)
        train_encodings = tokenizer(list(train_sequences),truncation = True, max_length = 150, return_tensors="pt")
        val_encodings = tokenizer(list(val_sequences),truncation = True, max_length = 150, return_tensors="pt")
        
        return train_encodings, val_encodings

    @staticmethod
    def model_setup(heavy_config:dict):
        model = SetupModel(heavy_config).model
        return model

    
    def add_dirs_to_train_args(self, params):
        result_dir, log_dir = self.setup_dirs(self.user_dir)
        params["logging_dir"] = log_dir
        params["output_dir"] = result_dir
        return params
        
    
    def train_args(self, params:dict):
        params = self.add_dirs_to_train_args(params)
        return TrainingArguments(**params)
        
    def setup_trainer(self, ):
        trainer = Trainer(
            model=self.model,                         # the instantiated 🤗 Transformers model to be trained
            args=self.train_params,                  # training arguments, defined above
            train_dataset=self.train_encodings,         # training dataset
            eval_dataset=self.val_encodings            # evaluation dataset
        )
        return trainer

    def train(self):
        self.trainer.train()



