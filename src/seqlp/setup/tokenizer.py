import gzip
import shutil
import pandas as pd
from .data_prep import Prepare
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
import os
    
class Batching:
    def __init__(self, tokenizer, batch_size = 10000):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
    

    def read_gzipped_csv(self, filename: str) :
        with gzip.open(filename, 'rt') as f:
            # Use chunksize to read in batches
            for chunk in pd.read_csv(f, dtype='string[pyarrow]', chunksize=self.batch_size):
                yield chunk.iloc[:, 0]
    
    def tokenize_batches(self,gz_filename, max_length):
        for batch in self.read_gzipped_csv(gz_filename):
            yield self.tokenizer(batch.tolist(), truncation=True, padding='max_length', max_length=max_length, return_tensors="pt") 
    

class TokenizeData:

    def __init__(self, pretrained_tokenizer = "facebook/esm2_t6_8M_UR50D", mlm_probability = 0.15) -> None:
        self.tokenizer = self.setup_tokenizer(pretrained_tokenizer)
        self.data_collator = self.masking(self.tokenizer, mlm_probability)


    @staticmethod
    def download_and_prepare(download_commands_script:str, limit = 1000000, save_single_csvs = False, user_dir = False, prep_data_type = "uniform") -> (str, str):            

        Prep = Prepare(download_commands_script, user_dir)
        concatenated_df:pd.Series = Prep.download_data(limit = limit, save_single_csvs = save_single_csvs, prep_data_type = prep_data_type)
        concatenated_df:pd.Series = Prep.drop_duplicates(concatenated_df)
        filename = os.path.join(Prep.save_dir, "concatenated.csv")
        train_sequences, val_sequences = Prep.create_train_test(concatenated_df)
        del concatenated_df
        train_filename = "train_"  + filename
        val_filename = "val_" + filename
        pd.Series(train_sequences).to_csv(train_filename, index = False)
        pd.Series(val_sequences).to_csv(val_filename, index = False)
        with open(train_filename, 'rb') as f_in:
            with gzip.open(train_filename + '.gz', 'wb') as f_out:
                  shutil.copyfileobj(f_in, f_out)
        os.remove(train_filename)
        with open(val_filename, 'rb') as f_in:
            with gzip.open(val_filename+ '.gz', 'wb') as f_out:
                  shutil.copyfileobj(f_in, f_out)
        os.remove(val_sequences)
        return (train_filename + '.gz', val_filename+ '.gz')
    
    def setup_tokenizer(self, pretrained_tokenizer:str):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer)
        return tokenizer

    def tokenize(self, gz_filename:str, max_length = 150,  batch_size = 10000):
        Batch = Batching(self.tokenizer, batch_size)
        encodings = Batch.tokenize_batches(gz_filename, max_length)
     #   val_labels = val_sequences["labels"].tolist()
        return encodings
    
    @staticmethod
    def masking(tokenizer, mlm_probability):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,  # Set to True for masked language modeling
            mlm_probability=mlm_probability  # Probability of masking a token
        )
        return data_collator

