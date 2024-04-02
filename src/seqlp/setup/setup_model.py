from transformers import DistilBertModel, DistilBertConfig

class SetupModel:

    def __init__(self, heavy_config, model_type = "distilBert"):

        self.model, self.config = getattr(self, model_type)(heavy_config)

    @staticmethod
    def distilBert(config):
        config = DistilBertConfig.from_dict(config)
        model = DistilBertModel(config)
        return model, config


# Initializing a model from the configuration

