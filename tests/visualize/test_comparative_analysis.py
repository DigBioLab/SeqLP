from src.seqlp.visualize.comparative_analysis import ExtractData, ComparativeAnalysis, TransformData, LoadModel
import numpy as np 


def test_ExtractData():
    test_path = r"C:/Users/nilsh/my_projects/ExpoSeq/my_experiments/max_new/sequencing_report.csv"
    test_head_no = 3
    sequencing_report = ExtractData().extract_from_csv(test_path, head_no = test_head_no)
    assert "full_sequence" in sequencing_report.columns, "The full sequence should be in the columns"
    
    
    
def test_ComparativeAnalysis():
    assert ComparativeAnalysis.reformat_positions([(1, 3), (5, 7)]) == [1, 2, 3, 5, 6, 7], "Reformat positions does not work as expected."
    assert ComparativeAnalysis.reformat_multi_sequence(np.array("A-B-C")) == "ABC", "Reformat multi sequence does not work as expected."
    Setup = LoadModel(model_path = r"C:\Users\nilsh\my_projects\SeqLP\tests\test_data\nanobody_model")
    assert ComparativeAnalysis._get_top_attentions(Setup, "ABC", [1, 2, 3], no_top_heads = 1).shape == (3,3), "The shape of the attention matrix should be symmetric and be the length of the sequence."
    arr = np.array([1, 2, 3, 4, 5])
    result = TransformData.normalize_and_standardize(arr)
    assert np.max(result) == 1.0, "The maximum value should be 1.0"