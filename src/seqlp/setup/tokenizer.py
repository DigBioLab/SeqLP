import gzip
import shutil
import pandas as pd
from .data_prep import Prepare
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
import os
    
class TokenizeData:

    def __init__(self) -> None:
        self.tokenizer = None
        self.data_collator = None

    @staticmethod
    def download_and_prepare(download_commands_script:str, limit = 1000000, save_single_csvs = False, user_dir = False, prep_data_type = "uniform") -> str:            

    
        Prep = Prepare(download_commands_script, user_dir)
        concatenated_df:pd.Series = Prep.download_data(limit = limit, save_single_csvs = save_single_csvs, prep_data_type = prep_data_type)
        concatenated_df:pd.Series = Prep.drop_duplicates(concatenated_df)
        filename = os.path.join(Prep.save_dir, "concatenated.csv")
        concatenated_df.to_csv(filename, index = False)
        with open(filename, 'rb') as f_in:
            with gzip.open(filename + '.gz', 'wb') as f_out:
                  shutil.copyfileobj(f_in, f_out)
        os.remove(filename)
        return filename + '.gz'
    
    
    def tokenize(self, gz_filename:str, max_length = 150, mlm_probability = 0.15):
        concatenated_df = Prepare.read_gzipped_csv(gz_filename)
        train_sequences, val_sequences = Prepare.create_train_test(concatenated_df)
        
        model_checkpoint = "facebook/esm2_t6_8M_UR50D"
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        train_encodings = self.tokenizer(train_sequences.tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")

        val_encodings = self.tokenizer(val_sequences.tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors="pt")

        self.data_collator = self.masking(self.tokenizer, mlm_probability)
     #   val_labels = val_sequences["labels"].tolist()
        return train_encodings, val_encodings,
    
    @staticmethod
    def masking(tokenizer, mlm_probability):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,  # Set to True for masked language modeling
            mlm_probability=mlm_probability  # Probability of masking a token
        )
        return data_collator

