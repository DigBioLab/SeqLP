import re
import torch
from transformers import AutoTokenizer, EsmModel
import os


class LoadModel:
    def __init__(self, model_path:str, model = None, tokenizer = None):
        if model == None and tokenizer == None:
            self.load_model(model_path)
        else:
            self.model = model
            self.tokenizer = tokenizer
    
    def load_model(self, model_path:str):
    #    assert os.path.isdir(model_path), "Your model path is not a directory. Please provide a directory with the model files. The files must be config.json, model.safetensors, training_args.bin"
      #  assert os.path.isfile(os.path.join(model_path, "model.safetensors")), "Your model path does not contain a model.safetensors file. Please provide a directory with the model files. The files must be config.json, model.safetensors, training_args.bin"
        self.model = EsmModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    def _get_encodings(self, sequence):
        assert type(sequence) == list, "The sequence must be a list of strings"
        assert len(sequence) > 0, "You need to provide at least one sequence"
        assert type(sequence[0]) == str, "The sequence must be a string"
        sequences = [" ".join(list(re.sub(r"[UZOB*_]", "X", sequence))) for sequence in sequence]
        encoded_input = self.tokenizer(sequences, return_tensors='pt', padding=True)
        return encoded_input
    def get_attention(self, sequence:list):
        """Returns attentions of the loaded model and removes the CLS and SEP tokens from those. It assumes that these are the first and last tokens in the sequence, respectively.

        Args:
            sequence (list): A list of sequences

        Returns:
            _type_: _description_
        """
        encoded_input = self._get_encodings(sequence)
        self.model.eval()
        tokens = encoded_input['input_ids']
        with torch.no_grad():
            output = self.model(**encoded_input, output_attentions=True)
        attentions = torch.stack(output['attentions'], dim=1)
                
        # Remove [CLS] and [SEP] token attentions
        # Assuming [CLS] at index 0 and [SEP] at the last index of each sequence
        seq_len = attentions.shape[3]  # Assuming all sequences are padded to the same length
        # Remove the first and last token (typically [CLS] and [SEP])
        attentions = attentions[:, :, :, 1:seq_len-1, 1:seq_len-1]
        return attentions