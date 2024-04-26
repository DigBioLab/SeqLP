from data_prep import FromTSV
import pandas as pd
from Bio import AlignIO
from Bio.Align import AlignInfo
import math
from Bio import SeqIO
import subprocess
import editdistance
import pandas as pd
import os

class Fasta:
    def create_fasta(sequences:pd.Series, output_file = 'sequences.fasta'):
        # Write the sequences to a FASTA file
        with open(output_file, 'w') as file:
            for index, sequence in sequences.items():
                # Write the header with the sequence identifier
                file.write(f'>{index}\n')
                # Write the sequence, can wrap this with textwrap for specific line lengths
                file.write(f'{sequence}\n')

class MSACluster:
    @staticmethod
    def make_seq_record_of_fasta(fasta_file:str) -> list:
        seq_records = []

        # Open the FASTA file and read each record
        with open(fasta_file, 'r') as fasta_file:
            for seq_record in SeqIO.parse(fasta_file, 'fasta'):
                seq_records.append(seq_record)
        return seq_records
    
    def run_msa(self,muscle_path, fasta_path, out = "aligned_sequences.fasta") -> list[str]:
        assert type(muscle_path) == str, "The muscle_path should be a string"
        assert muscle_path.endswith(".exe"), "The muscle_path should end with .exe"
        if not os.path.isfile(muscle_path):
            raise FileNotFoundError(f"The muscle_path file {muscle_path} does not exist.")       
        subprocess.run([muscle_path, "-in", fasta_path, "-out", out])
        alignment = AlignIO.read(out, 'fasta')
        sequence_strings = []
        for record in alignment:
            # Convert the Seq object to a string and append to the list
            sequence_strings.append(str(record.seq))
        return sequence_strings
    
    @staticmethod
    def calculate_distance(seq1, seq2):
        """Calculate the edit distance between two sequences, including gap penalties."""
        return editdistance.distance(seq1, seq2)
    
    def distance_on_gaps(self, sequence_strings:list, max_distance_percent = 0.2) -> list[list]:
        assert type(max_distance_percent) == float, "The max_distance_percent should be a float"
        assert max_distance_percent < 1, "The max_distance_percent should be less than 1"
        assert max_distance_percent > 0, "The max_distance_percent should be greater than 0"
        max_distance = round(len(sequence_strings[0]) * max_distance_percent)
        clusters = []
        # Iterate over each sequence
        for seq in sequence_strings:
            # Try to find a cluster for the sequence
            found_cluster = False
            for cluster in clusters:
                # Check if the sequence is close enough to any element in the cluster
                if any(editdistance.distance(seq, member) <= max_distance for member in cluster):
                    cluster.append(seq)
                    found_cluster = True
                    break
            # If no suitable cluster is found, start a new cluster
            if not found_cluster:
                clusters.append([seq])
        return clusters
        
    @staticmethod
    def find_variable_positions(clusters:list[list]) -> list[tuple]:
        """Finds the variable positions in the cluster of sequences. 

        Args:
            clusters (list[list]): Clusters found by the distance_on_gaps function. Number of clusters equals the number of items in the first list. 
            Clusters contain multiple sequence aligned sequences, so the sequences have all the same length.

        Returns:
            list[tuple]: Number of tuples equals the number of clusters. First item is the list of sequences and second item is the list of variable positions
        """
        variable_positions = []
        for cluster in clusters:
            if len(cluster) == 1:  # If there's only one sequence, no variability
                variable_positions.append((cluster, []))
                continue
            
            # Initialize a set to keep track of varying positions
            varying_indexes = set()
            
            # Compare each sequence with every other sequence in the cluster
            reference_sequence = cluster[0]
            sequence_length = len(reference_sequence)
            
            for i in range(sequence_length):
                reference_char = reference_sequence[i]
                for sequence in cluster[1:]:
                    if sequence[i] != reference_char:
                        varying_indexes.add(i)
                        break
            
            # Store the result as a tuple of the cluster and the sorted list of varying positions
            variable_positions.append((cluster, sorted(varying_indexes)))
        
        return variable_positions
    
    def find_fixed_positions(self, clusters:list[list]) -> list[tuple]:
        variable_positions_sequences: list[tuple] = self.find_variable_positions(clusters)
        
        for index, cluster in enumerate(variable_positions_sequences):
            example_sequence = cluster[0][0]
            variable_positions_cluster = cluster[1]
            all_positions = list(range(len(example_sequence)))
            fixed_positions = [pos for pos in all_positions if pos not in variable_positions_cluster]
            variable_positions_sequences[index] = (cluster[0], fixed_positions)
        return variable_positions_sequences
    
    @staticmethod
    def translate_positions(cluster):
        gapped_sequence = cluster[0][0]
        positions = cluster[1]
        ungapped_sequence = gapped_sequence.replace('-', '')  # Remove gaps
        mapping = []  # This will map gapped indices to ungapped indices
        ungapped_index = 0
        
        for index, char in enumerate(gapped_sequence):
            if char != '-':
                mapping.append(ungapped_index)
                ungapped_index += 1
            else:
                mapping.append(None)  # No corresponding index in ungapped sequence

        translated_positions = [mapping[pos] for pos in positions if mapping[pos] is not None]
        return ungapped_sequence, translated_positions
            
    @staticmethod
    def get_representative_sequence(ungapped_sequences):
        return ungapped_sequences[0]
    
    
    
muscle_path = r"C:\Users\nilsh\Downloads\muscle3.8.31_i86win32.exe"
sequences_series = FromTSV.read_gzip_tsv()
print("You will align the following number of sequences:" + str(sequence_series.shape[0]))
Fasta.create_fasta(sequence_series, "nanobody_sequences.fasta")
aligned_sequences = MSACluster().run_msa(muscle_path, "nanobody_sequences.fasta", "aligned_sequences.fasta")
# Create a summary object
alignment = AlignIO.read("aligned_sequences.fasta", "fasta")
summary_align = AlignInfo.SummaryInfo(alignment)


thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
# Calculate consensus sequence (simple approach to find conserved residues)
threshold_seq = {}
score_matrices = {}
for threshold in thresholds:
    consensus = summary_align.dumb_consensus(threshold=threshold)  # 80% consensuss
    
    pssm = summary_align.pos_specific_score_matrix(consensus, chars_to_ignore=['-'])
    threshold_seq[threshold] = str(consensus)
    entropy = [sum(-p * math.log(p) for p in col.values() if p > 0) for col in pssm]
    score_matrices[threshold]  = entropy
    print(f"Consensus sequence showing conserved residues at {threshold * 100}% threshold:")
    print(consensus)

import json 

with open("consensus.json", "w") as file:
    json.dump(threshold_seq, file)

with open("entropy.json", "w") as file:
    json.dump(score_matrices, file)
