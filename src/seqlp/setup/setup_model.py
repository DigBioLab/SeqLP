


# oriented on: https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling
class SetupModel:

    def __init__(self, heavy_config, model_type = "distilBert"):
        
        self.model, self.config = getattr(self, model_type)(heavy_config)

    @staticmethod
    def distilBert(config):
        from transformers import DistilBertForMaskedLM, DistilBertConfig
        config = DistilBertConfig.from_dict(config)
        model = DistilBertForMaskedLM(config)
        return model, config
    
    @staticmethod
    def esm(config):
        from transformers import EsmForMaskedLM, EsmConfig
        config = EsmConfig.from_dict(config)
        model = EsmForMaskedLM(config)
        return model, config

    @staticmethod
    def roberta(config):
        from transformers import RobertaForMaskedLM, RobertaConfig
        config = RobertaConfig.from_dict(config)
        model = RobertaForMaskedLM(config)
        return model, config
    
    @staticmethod
    def bert(config):
        from transformers import BertForMaskedLM, BertConfig
        config = BertConfig.from_dict(config)
        model = BertForMaskedLM(config)
        return model, config
    
    @staticmethod
    def bart(config):
        from transformers import BartForConditionalGeneration, BartConfig
        config = BartConfig.from_dict(config)
        model = BartForConditionalGeneration(config)
        return model, config
    
    def mpnet(config):
        from transformers import MPNetForMaskedLM, MPNetConfig
        config = MPNetConfig.from_dict(config)
        model = MPNetForMaskedLM(config)
        return model, config
    
        

# Initializing a model from the configuration

