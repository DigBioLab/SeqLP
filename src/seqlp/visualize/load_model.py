import re
import torch
from transformers import AutoTokenizer, EsmModel, AutoModel
import os
import torch.nn.functional as F


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
        self.model = AutoModel.from_pretrained(model_path)
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
    
    
    def _get_perplexity(self, sequence: str) -> float:
        # Encode the input sequence
        encoded_input = self._get_encodings([sequence])
        input_ids = encoded_input['input_ids']

        # Ensure input_ids has a batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Adds a batch dimension
        
        # Create a mask for random tokens to be predicted (15% probability)
        mask = torch.rand(input_ids.shape) < 0.15
        masked_indices = mask.nonzero(as_tuple=True)
        
        # Check if any token is masked, and handle the case where no tokens are masked
        if masked_indices[0].numel() == 0:
            return float('inf')  # Return infinity if no token is masked, as perplexity can't be calculated

        # Store original tokens at masked positions
        original_tokens = input_ids[masked_indices].clone()

        # Replace masked token ids with the mask token id
        input_ids[masked_indices] = self.tokenizer.mask_token_id
        
        # Calculate the model output without calculating gradients
        with torch.no_grad():
            outputs = self.model(input_ids)
            last_hidden_state = outputs.last_hidden_state
            # Project last hidden state to vocabulary space using the embedding matrix
            logits = self.model.embeddings.word_embeddings.weight.matmul(last_hidden_state.transpose(1, 2)).transpose(1, 2)

        # Calculate loss manually using cross-entropy at the positions of the masked tokens
        # Only calculate loss for masked positions, reshaping as necessary for cross-entropy
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), original_tokens.view(-1), reduction='sum')
        
        # Calculate and return perplexity
        perplexity = torch.exp(loss / masked_indices[0].numel())
        return perplexity
        
    
