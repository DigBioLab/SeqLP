from .msa_cluster import MSACluster, Fasta
import pandas as pd
from .load_model import LoadModel, ExtractData, TransformData
import numpy as np 
from .bertology import Bertology, AttentionAnalysis
import os



class ComparativeAnalysis:
    @staticmethod
    def aling_to_each_other(sequencing_report, muscle_path) -> np.array:
        if not os.path.isfile(muscle_path):
            raise FileNotFoundError("The muscle path is not correct")
        full_sequences = sequencing_report['full_sequence']
        Fasta.create_fasta(full_sequences, "nanobody_sequences.fasta")
        aligned_sequences = MSACluster().run_msa(muscle_path, "nanobody_sequences.fasta", "aligned_sequences.fasta")
        seq_array = np.array([list(seq) for seq in aligned_sequences])
        return seq_array
    @staticmethod
    def reformat_positions(position:list) -> list:
        return [i for start, end in position for i in range(start, end + 1)]
    
    @staticmethod
    def _get_top_attentions(Setup, sequence:str, all_positions:list, no_top_heads:int = 5):
        attentions = Setup.get_attention(sequence = [sequence])
        assert type(all_positions[0]) == int
        Berto = Bertology(residues = all_positions, sequence = sequence, decision_function = "binding_site")
        pa_f = Berto.compute_pa_f_fast(attentions)
        heads_top_mean = AttentionAnalysis().calculate_top_head_one(pa_f, no_top_heads, attentions, method = "mean")
        return heads_top_mean
    
    @staticmethod
    def reformat_multi_sequence(sequence:np.array):
        sequence = "".join(sequence.flatten().astype(str))
        sequence = sequence.replace("-", "")
        return sequence


    def loop_sequences_and_align(self, sequencing_report:pd.DataFrame,muscle_path:str,Setup, normalization = True):
        seq_array = self.aling_to_each_other(sequencing_report, muscle_path=muscle_path)
        collection_top_heads_mean = []
        mask_array = (seq_array != '-').astype(int) # numpy array where dashes are 0. 
        positions = sequencing_report['CDRPositions']
        for sequence, position, row in zip(seq_array, positions, range(mask_array.shape[0])):
            all_positions = self.reformat_positions(position)
            sequence = self.reformat_multi_sequence(sequence)
            mask_row = mask_array[row, :]
            heads_top_mean = self._get_top_attentions(Setup, sequence, all_positions, no_top_heads = 5)
            remove_i =0
            remove_j = 0
            final_matrix = np.zeros((mask_row.shape[0], mask_row.shape[0]))
            for i in range(mask_row.shape[0]): # extend the heads matrix to the multiple sequence alignmnet length 
                i_value = mask_row[i]
                if i_value == 0:
                    remove_i += 1
                    continue
                for j in range(mask_row.shape[0]):
                    j_value = mask_row[j]
                    if j_value == 1:
                        value = heads_top_mean[i-remove_i, j - remove_j]
                        final_matrix[i,j] = value
                    else:
                        remove_j += 1
                remove_j = 0
            if normalization == True:
                pdf_weights = TransformData.normalize_and_standardize(final_matrix)
            else:
                pdf_weights = final_matrix
            collection_top_heads_mean.append(pdf_weights)
        return collection_top_heads_mean
    
    


class Pipeline:
    def __init__(self, model_path:str):
        self.Setup = LoadModel(model_path = model_path)
        self.GetData = ExtractData()
        self.Compare = ComparativeAnalysis()
        
    def run_pipeline(self, path_csv, number_sequences_per_group, muscle_path) -> np.array: # shape no_sequence x multi sequence alignment length x multi sequence laignment length    
        sequencing_report = self.GetData.extract_from_csv(path_csv, head_no = number_sequences_per_group)
        collection_top_heads_mean = self.Compare.loop_sequences_and_align(sequencing_report, muscle_path, self.Setup, normalization = True)
        collection_array = np.stack(collection_top_heads_mean, axis = 0)
        assert collection_array.shape[0] == sequencing_report.shape[0], "The number of sequences should match the number of sequences in the collection array"
        return collection_array
    
    def single_sequence(self, clone):
        full_sequence, cdr_positions = self.GetData.calculate_cdr_positions(clone)
        all_positions = self.Compare.reformat_positions(cdr_positions)
        heads_top_mean = self._get_top_attentions(self.Setup, full_sequence, all_positions, no_top_heads = 5)
        normalized_array = TransformData.normalize_and_standardize(heads_top_mean)
        return normalized_array
            
            




#collection_array = run_pipeline(r"C:/Users/nilsh/my_projects/ExpoSeq/my_experiments/max_new/sequencing_report.csv", 3, r"C:\Users\nilsh\Downloads\muscle3.8.31_i86win32.exe")
