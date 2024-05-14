import os
import pandas as pd
from src.seqlp.setup.data_prep import FromTSV, Prepare


sequence_series = FromTSV.read_gzip_tsv(r"c:\Users\nilsh\OneDrive\Desktop\results_thesis\data\training_data\ngs_sequence.tsv.gz", limit = 20000000)
sequence_series = Prepare.drop_duplicates(sequence_series)
sequence_series = Prepare.insert_space(sequence_series) # very important otherwise the tokenizing will not work appropriately.
train_sequences, val_sequences = Prepare.create_train_test(sequence_series)
print(train_sequences.shape, val_sequences.shape)
train_sequences.to_csv(r"C:\Users\nilsh\OneDrive\Desktop\results_thesis\data\training_data\train.csv", index = False)
val_sequences.to_csv(r"C:\Users\nilsh\OneDrive\Desktop\results_thesis\data\training_data\val.csv", index = False)
