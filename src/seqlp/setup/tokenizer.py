from transformers import PreTrainedTokenizerFast
from transformers import BertTokenizer
class AminoAcidTokenizer(PreTrainedTokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(
            # Define special tokens
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
            **kwargs
        )
        
        # Amino acid single-letter codes as tokens
        self.amino_acid_vocab = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", 
                                 "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        
        # Add special tokens to the vocab
        self.vocab = {**{v: k for k, v in enumerate(self.amino_acid_vocab)},
                      **self.added_tokens_encoder}
        
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}



