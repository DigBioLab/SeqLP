from .msa_cluster import MSACluster, Fasta
import pandas as pd
from .load_model import LoadModel
import numpy as np 
from .bertology import Bertology, AttentionAnalysis
import os

class TransformData:
    @staticmethod
    def normalize_and_standardize(arr:np.array) -> np.array:
        max = np.max(arr)        
        # Standardize the array to sum to 1
        if max == 0:
            normalized_array = arr  # or handle as needed, e.g., set to zero or leave unchanged
        else:
            # Normalize the array
            normalized_array = arr / max
        return normalized_array

    def hellinger_distance(p, q):
        p = p.flatten()
        q = q.flatten()
        return (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2))



class ExtractData:
    @staticmethod
    def calculate_cdr_positions(row):
        full_sequence = ''.join(row)
        running_length = 0
        cdr_positions = []

        for key in row.index:
            current_length = len(row[key])
            if 'CDR' in key:  # Check if the region is a CDR

                cdr_positions.append((running_length, running_length + current_length - 1))
            running_length += current_length
        
        return full_sequence, cdr_positions
    
    def extract_full_sequence_from_regions(self, sequencing_report:pd.DataFrame) -> pd.DataFrame:
        """Generates a full sequence in a separate column and gets the CDR positions which are necessary for constraining. 
        Alignment of the sequences is necessary for the comparative analysis.

        Args:
            sequencing_report (pd.DataFrame): sequencing_report with aligned fragments

        Returns:
            pd.DataFrame: sequencing report with full sequence and cdr postions
        """
        if any(column not in sequencing_report.columns for column in ["aaSeqCDR1","aaSeqFR2","aaSeqCDR2","aaSeqFR3","aaSeqCDR3","aaSeqFR4"]):
            raise ValueError("The columns should contain the regions of the nanobody")
        sequencing_report = sequencing_report[["aaSeqCDR1","aaSeqFR2","aaSeqCDR2","aaSeqFR3","aaSeqCDR3","aaSeqFR4"]]
        sequencing_report[['full_sequence', 'CDRPositions']] = sequencing_report.apply(self.calculate_cdr_positions, axis=1, result_type='expand')
        return sequencing_report
    
    def extract_from_csv(self, path , head_no = 3):

        assert path.endswith(".csv"), "The path should be a csv file"
        sequencing_report = pd.read_csv(path)
        if "Experient" in sequencing_report.columns:
            sequencing_report = sequencing_report.groupby("Experiment").head(head_no)
            experiments = sequencing_report['Experiment'].tolist()
        else:
            sequencing_report = sequencing_report.head(head_no)
        sequencing_report = self.extract_full_sequence_from_regions(sequencing_report)
        return sequencing_report




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
    
    


    
def run_pipeline(path_csv, number_sequences_per_group, muscle_path,  ) -> np.array: # shape no_sequence x multi sequence alignment length x multi sequence laignment length    
    Setup = LoadModel(model_path = r"C:\Users\nilsh\my_projects\SeqLP\tests\test_data\nanobody_model")
    GetData = ExtractData()
    sequencing_report = GetData.extract_from_csv(path_csv, head_no = number_sequences_per_group)
    Compare = ComparativeAnalysis()
    collection_top_heads_mean = Compare.loop_sequences_and_align(sequencing_report, muscle_path, Setup, normalization = True)
    collection_array = np.stack(collection_top_heads_mean, axis = 0)
    assert collection_array.shape[0] == sequencing_report.shape[0], "The number of sequences should match the number of sequences in the collection array"
    return collection_array



#collection_array = run_pipeline(r"C:/Users/nilsh/my_projects/ExpoSeq/my_experiments/max_new/sequencing_report.csv", 3, r"C:\Users\nilsh\Downloads\muscle3.8.31_i86win32.exe")
