import requests
import pandas as pd
import glob
import os
import gzip
import shutil
from sklearn.model_selection import train_test_split
import numpy as np 
import itertools
import random

class DataHandler:
   
   def __init__(self, path_to_scripts, **kwargs):
      """_summary_

      Args:
          path_to_scripts (str): You can parse None but then you should parse dir_main and the directory
      """
      self.path_to_scripts = path_to_scripts
      self.train_data_dir = self.setup_data_dir(path_to_scripts)
      
      

   
   def chooseIg(self, ig_type = "BULK"):
      """chooses the files based on the prefered IG setting. 

      Args:
          ig_type (str, optional): Defaults to "BULK".

      Returns:
          list: List with paths to the zipped csv files that contain the training information
      """
      assert ig_type in ["BULK","IGHA", "IGHM", "IGD", "IGHG", "IGHE"], "The ig_type you parsed is not valid. Please choose from BULK, IGHM, IGHG, IGHE, IGHM, IGD"
      all_zipped_csvs = glob.glob(self.train_data_dir + "\*")
      filtered_zipped_csvs = [file for file in all_zipped_csvs if ig_type in file]
      return filtered_zipped_csvs
   
   def delete_rest(self, filtered_zipped_csvs:list):
      """Deletes all other files based on the chosen ig_type in chooseIg

      Args:
          filtered_zipped_csvs (list): output from chooseIg
      """
      all_zipped_csvs = glob.glob(self.train_data_dir + "\*")
      for file in all_zipped_csvs:
         if file not in filtered_zipped_csvs:
            os.remove(file)
            
   
            
class ReadCSV:
   
   
   def __init__(self) -> None:
      self.cols_of_interest = ["fwr1_aa", "cdr1_aa", "fwr2_aa","cdr2_aa",  "fwr3_aa", "cdr3_aa", "fwr4_aa", "sequence_alignment_aa"]
      self.concatenated_df = pd.Series(dtype='string[pyarrow]')
      self.csv_filename = ""
   
   def unzip_and_read_csv(self, filename,) -> pd.DataFrame:

      with gzip.open(filename, 'rt') as f:
            aa_alignment_sequence = pd.read_csv(f, usecols = self.cols_of_interest, skiprows = 1, )
      return aa_alignment_sequence
   
   def get_sequence_num(self):
      return self.concatenated_df.shape[0]
   
   
   def concatenate(self, aa_alignment_sequence:pd.DataFrame):
      self.concatenated_df = pd.concat([self.concatenated_df, aa_alignment_sequence])
      

      

      
class Prepare:
   def __init__(self, path, user_dir = None):
      self.save_dir = self.setup_data_dir(path, user_dir)
      self.path_to_scripts = path
      self.CSVSaver = ReadCSV()
      
   @staticmethod
   def setup_data_dir(path_to_scripts = None, user_dir = None ) -> str:
      if user_dir == None:
         dir_main = os.getcwd()
      else:
         dir_main = user_dir
      if not os.path.isdir(os.path.join(dir_main, "train_data")):
         os.mkdir(os.path.join(dir_main, "train_data"))
      return os.path.join(dir_main, "train_data")
   
   @staticmethod
   def input_check(prep_data_type, limit, sh_scripts):
      assert len(sh_scripts) > 0, "No files found in the directory"
      assert limit > 1, "The limit has to be greater than 1"
      
      
   def download_data(self, limit = 10, save_single_csvs = False, prep_data_type = "full_sequence") -> pd.Series:
      """Downloads the data in the file with the wget commands and filters the columns in the csvs and saves them again as gzipped csvs in self.save_dir.
      """
      data_prep_type = {"uniform": "create_uniform_series_with_random_length",
               "fragment_directed": "create_uniform_series_chain_specific",
               "full_sequence": "create_series_full_nanobody"}
      if prep_data_type not in data_prep_type.keys():
         raise ValueError("The data preparation type is not valid. Please choose from uniform or fragment_directed")
      assert os.path.isdir(self.path_to_scripts), "The path to the scripts is not valid. Your path is: " + self.path_to_scripts + " and not a directory."
      sh_scripts = glob.glob(self.path_to_scripts + "/*.sh")
      self.input_check(prep_data_type, limit, sh_scripts)
      for sh_script in sh_scripts:
         with open(sh_script, "r") as f:
            for line in f:
               try:
                  if line.startswith('wget'):
                     url = line.split(' ')[1].strip()
                     name = os.path.basename(url).split(".")[0]
                     save_name = os.path.join(self.save_dir, name)
                     response = requests.get(url)
                     # Make sure the request was successful
                     response.raise_for_status()
                     # Write the content of the response to a file
                     with open(save_name, 'wb') as out_file:
                        out_file.write(response.content)
                        
                     aa_alignment_sequence = self.CSVSaver.unzip_and_read_csv(save_name)
                     sequence_series = getattr(self, data_prep_type[prep_data_type])(aa_alignment_sequence)
                     sequence_series = self.insert_space(sequence_series)

                     self.CSVSaver.concatenate(sequence_series)
                     os.remove(save_name)


                  if self.CSVSaver.get_sequence_num() > limit:
                     break
               except:
                  pass
      return self.CSVSaver.concatenated_df

   def create_uniform_series_chain_specific(self, concatenated_df: pd.DataFrame) -> pd.Series:
      """This function takes a subset of columns (regions) and merges the sequential parts together so that in the final series are sequences of different length from the heavy chain.

      Args:
          concatenated_df (pd.DataFrame): contains the regions and full sequence in a dataframe

      Returns:
          pd.Series: one column with different sequence length from heavy chain.
      """
      cols_of_interest = self.CSVSaver.cols_of_interest
      full_sequence_col = "sequence_alignment_aa"

      # generate all possible sequential combinations of the fragments, excluding the full sequence
      fragment_cols = [col for col in cols_of_interest if col != full_sequence_col]
      combinations = []
      for r in range(1, len(fragment_cols) + 1):
         combinations.extend(itertools.combinations(fragment_cols, r))

      # filter out non-sequential combinations
      combinations = [c for c in combinations if all((c[i+1] == fragment_cols[fragment_cols.index(c[i])+1]) for i in range(len(c)-1))]

      # add the full sequence as a standalone option
      combinations.append((full_sequence_col,))

      sequences = []
      for _, row in concatenated_df.iterrows():
         # randomly select one combination
         combination = random.choice(combinations)
         valid_combination = [col for col in combination if pd.notna(row[col])] # removes columns if it has nan
         # concatenate the corresponding columns
         sequence = ''.join(row[col] for col in valid_combination)
         sequences.append(sequence)

      return pd.Series(sequences, name='sequences', dtype='string[pyarrow]')
   @staticmethod
   def create_series_full_nanobody(concatenated_df: pd.DataFrame) -> pd.Series:
      """This function just takes the full sequence from the heavy chain.

      Args:
          concatenated_df (pd.DataFrame): contains the regions and full sequence in a dataframe

      Returns:
          pd.Series: one column with the full sequence from heavy chain.
      """
      sequences = concatenated_df['sequence_alignment_aa'].tolist()
      return pd.Series(sequences, name='sequences', dtype='string[pyarrow]')
   
   @staticmethod
   def create_uniform_series_with_random_length(concatenated_df: pd.DataFrame, minimum_length = 10) -> pd.Series:
      """This function just takes a random part of the full sequence if it is longer than 5, otherwise it takes the full sequence.

      Args:
          concatenated_df (pd.DataFrame): contains the regions and full sequence in a dataframe

      Returns:
          pd.Series: one column with different sequence length from heavy chain.
      """
      sequences = []
      max_length = concatenated_df['sequence_alignment_aa'].str.len().max()
      all_lengths = list(range(minimum_length, max_length + 1))
      np.random.shuffle(all_lengths)
      length_index = 0
      for _, row in concatenated_df.iterrows():
         full_sequence = row['sequence_alignment_aa']
         if len(full_sequence) > 5:
               while all_lengths[length_index] > len(full_sequence):
                  length_index += 1
                  if length_index >= len(all_lengths):
                     np.random.shuffle(all_lengths)
                     length_index = 0
               length = all_lengths[length_index]
               length_index += 1
               if length_index >= len(all_lengths):
                  np.random.shuffle(all_lengths)
                  length_index = 0
               start = np.random.randint(0, len(full_sequence) - length + 1)  # generate a random start point
               sequence = full_sequence[start:start+length]
         else:
               sequence = full_sequence
         sequences.append(sequence)
      return pd.Series(sequences, name='sequences', dtype='string[pyarrow]')
      
   @staticmethod
   def drop_duplicates(concatenated_series:pd.Series) -> pd.Series:
      return concatenated_series.dropna().drop_duplicates()
   
   @staticmethod
   def create_train_test(sequences:pd.Series):
      train_sequences, val_sequences = train_test_split(sequences, test_size=0.2, random_state=42)
      return train_sequences, val_sequences

   @staticmethod
   def insert_space(concatenated_df:pd.Series) -> pd.Series:
      """Inserts a space between the amino acids.

      Args:
         concatenated_df (pd.sSeries): _description_

      Returns:
         pd.DataFrame: _description_
      """
      concatenated_df = concatenated_df.dropna()
      seqs = concatenated_df.values.tolist()
      assert isinstance(concatenated_df, pd.Series)
      concatenated_df = concatenated_df.apply(lambda x: ' '.join(list(x)) if isinstance(x, str) else x)
      return concatenated_df

   @staticmethod
   def read_gzipped_csv(filename:str) -> pd.Series:
      with gzip.open(filename, 'rt') as f:
         df = pd.read_csv(f, dtype='string[pyarrow]')
      serie = df.iloc[:, 0]
      return serie
   
   
   
   
   
import os
import pandas as pd


#download_commands_script = r"/zhome/20/8/175218/SeqLP/tests/test_data"
#if not os.path.isdir("/test_tmp"):
 ##   os.mkdir("/test_tmp")
#user_dir = "/zhome/20/8/175218/NLP_train/test_launch/test"
#limit = 2
#save_single_csvs = False
#prep_data_type = "full_sequence"
#Prep = Prepare(download_commands_script, user_dir)
#concatenated_df:pd.Series = Prep.download_data(limit = limit, save_single_csvs = save_single_csvs, prep_data_type = prep_data_type)