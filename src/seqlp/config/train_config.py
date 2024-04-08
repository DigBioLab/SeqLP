import os
import json

class SetupTrainConfig:
    def __init__(self) -> dict:
        dir = os.path.dirname(__file__)
        self.config = self.read_json(os.path.join(dir, "default_config_train.json"))
        
    @staticmethod
    def read_json(filepath):
        assert os.path.isfile(filepath), "Please provide the correct filepath to the model config file."
        with open(filepath, 'r') as json_file:
            data = json.load(json_file)
        return data

        
    def get_config(self,**kwargs:dict) ->dict:

        for key, value in kwargs.items():
            self.config[key] = value
        return self.config