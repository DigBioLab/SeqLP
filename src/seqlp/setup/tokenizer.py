import gzip
import shutil
import pandas as pd
from .data_prep import Prepare
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
import os
    
class TokenizeData:

    def __init__(self) -> None:
        self.tokenizer = None
        self.data_collator = None

    @staticmethod
    def download_and_prepare(download_commands_script:str, limit = 1000000, save_single_csvs = False, user_dir = False, prep_data_type = "uniform") -> str:            
        data_prep_type = {"uniform": "create_uniform_series_with_random_length",
                        "fragment_directed": "create_uniform_series_chain_specific",}
    
        Prep = Prepare(download_commands_script, user_dir)
        concatenated_df = Prep.download_data(limit = limit, save_single_csvs = save_single_csvs)
        concatenated_df = getattr(Prep, data_prep_type[prep_data_type])(concatenated_df)
        concatenated_df = Prep.drop_duplicates(concatenated_df)
        filename = os.path.join(Prep.save_dir, "concatenated.csv")
        concatenated_df.to_csv(filename, index = False)
        with open(filename, 'rb') as f_in:
            with gzip.open(filename + '.gz', 'wb') as f_out:
                  shutil.copyfileobj(f_in, f_out)
        os.remove(filename)
        return filename + '.gz'
    
    
    def tokenize(self, gz_filename:str):
        concatenated_df:pd.DataFrame = Prepare.read_gzipped_csv(gz_filename)
        concatenated_df = Prepare.insert_space(concatenated_df)
        train_sequences, val_sequences = Prepare.create_train_test(concatenated_df)
        
        model_checkpoint = "facebook/esm2_t6_8M_UR50D"
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        train_encodings = self.tokenizer(train_sequences["sequences"].tolist(), truncation=True, padding='max_length', max_length=150, return_tensors="pt")

        val_encodings = self.tokenizer(val_sequences["sequences"].tolist(), truncation=True, padding='max_length', max_length=150, return_tensors="pt")

        self.data_collator = self.masking(self.tokenizer)
     #   val_labels = val_sequences["labels"].tolist()
        return train_encodings, val_encodings,
    
    @staticmethod
    def masking(tokenizer):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,  # Set to True for masked language modeling
            mlm_probability=0.15  # Probability of masking a token
        )
        return data_collator

