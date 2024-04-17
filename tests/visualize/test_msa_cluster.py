from src.seqlp.visualize.msa_cluster import MSACluster
import os


def test_run_msa():
    file = os.path.abspath(r"tests/test_data/aligned_sequences.fasta")
    assert os.path.isfile(file) == True, "{file} does not exist"
    aligned_sequences = MSACluster().run_msa(r"C:\Users\nilsh\Downloads\muscle3.8.31_i86win32.exe", file)
    assert type(aligned_sequences) == list, "The aligned sequences should be a list of strings"
    assert len(aligned_sequences) == 18, "The number of aligned sequences should be 18"
    
def test_distance_on_gaps():
    file = os.path.abspath(r"tests/test_data/aligned_sequences.fasta")
    assert os.path.isfile(file), "{file} does not exist"
    aligned_sequences = MSACluster().run_msa(r"C:\Users\nilsh\Downloads\muscle3.8.31_i86win32.exe",file)
    clusters = MSACluster().distance_on_gaps(aligned_sequences, max_distance = 3)
    assert len(clusters) == 7, "The number of clusters should be 3"
    
def test_find_variable_and_fixed_positions():
    file = os.path.abspath(r"tests/test_data/aligned_sequences.fasta")
    assert os.path.isfile(file), "{file} does not exist"
    aligned_sequences = MSACluster().run_msa(r"C:\Users\nilsh\Downloads\muscle3.8.31_i86win32.exe", file)
    clusters = MSACluster().distance_on_gaps(aligned_sequences, max_distance = 3)
    variable_positions = MSACluster.find_variable_positions(clusters)
    assert type(variable_positions[0]) == tuple
    assert variable_positions[3][1] == [5, 7, 13, 14, 18, 24, 25], f"The variable positions are {variable_positions[3][1]}"
    assert len(variable_positions[0][0]) == 1, "there is only one sequence in this cluster"
    assert len(variable_positions[0][1]) == 0, "there is only one sequence in this cluster, so you cannot find variable positions"
    fixed_positions = MSACluster().find_fixed_positions(clusters)
    assert not any(elem in fixed_positions[3][1] for elem in [5, 7, 13, 14, 18, 24, 25]), "The fixed positions should not be the variable positions"