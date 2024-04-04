from setup.tokenizer import TokenizeData
import json
import os
import argparse
from setup.setup_model import SetupModel

class SetupModelConfig:
        
    def __init__(self) -> dict:
        dir = os.path.dirname(__file__)
        self.config = self.read_json(os.path.join(dir, "default_config_model.json"))
        
    def read_json(self, filepath):
        self.check_filepath(filepath)
        with open(filepath, 'r') as json_file:
            data = json.load(json_file)
        return data
    
    @staticmethod
    def check_filepath(filepath):
        assert os.path.isfile(filepath), "Please provide the correct filepath to the model config file."
    
    def set_tokenize_vocab(self, tokenize:TokenizeData):
        self.config["vocab_size"] = len(tokenize.tokenizer)
    
    def set_max_len(self, max_len:int):
        self.config["max_length"] = max_len
    
    def get_config(self, tokenize:TokenizeData, **kwargs:dict) -> dict:
        self.set_tokenize_vocab(tokenize)
        for key, value in kwargs.items():
            self.config[key] = value

        return self.config
    
    
