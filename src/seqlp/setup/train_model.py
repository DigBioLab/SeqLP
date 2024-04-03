from transformers import Trainer, TrainingArguments
from .setup_model import SetupModel
import os

from torch.utils.data import Dataset

class AminoAcidDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}



class TrainModel:
    heavy_config = {
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "hidden_size": 768,
        "d_ff": 3072,
        "vocab_size": 33,
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
    
    def __init__(self, train_dataset, validation_dataset, model_config = heavy_config,train_params = params, ):
        self.model = self.model_setup(heavy_config = model_config)
        self.train_params = self.train_args(train_params)
        self.trainer = self.setup_trainer(train_dataset, validation_dataset)
        
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
        
    def setup_trainer(self, train_encodings, val_encodings, data_collator):
        train_dataset = AminoAcidDataset(train_encodings)
        val_dataset = AminoAcidDataset(val_encodings)
        trainer = Trainer(
            model=self.model,                         # the instantiated 🤗 Transformers model to be trained
            args=self.train_params,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,
            data_collator=data_collator
            # evaluation dataset
        )
        return trainer

    def train(self):
        self.trainer.train()
        
        
