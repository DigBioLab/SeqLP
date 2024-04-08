from transformers import Trainer, TrainingArguments
from .setup_model import SetupModel
import os
from config.model_config import SetupModelConfig
from config.train_config import SetupTrainConfig
from torch.utils.data import Dataset
import torch

class AminoAcidDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}




class TrainModel:
    
    heavy_config = SetupModelConfig().config

    params = SetupTrainConfig().config

    def __init__(self, train_dataset, validation_dataset, data_collator, model_config = heavy_config,train_params = params, model_type = "distilBert", user_dir = None):
        self.user_dir = user_dir
        self.model = self.model_setup(heavy_config = model_config, model_type=model_type)
        self.train_params = self.train_args(train_params)
        self.trainer = self.setup_trainer(train_dataset, validation_dataset, data_collator)
        
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
    def model_setup(heavy_config:dict, model_type:str):
        model = SetupModel(heavy_config, model_type).model
        return model

    
    def add_dirs_to_train_args(self, params):
        result_dir, log_dir = self.setup_dirs(self.user_dir)
        params["logging_dir"] = log_dir
        params["output_dir"] = result_dir
        return params
        
    
    def train_args(self, params:dict):
        params = self.add_dirs_to_train_args(params)
        return TrainingArguments( **params)
        
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
        
        
