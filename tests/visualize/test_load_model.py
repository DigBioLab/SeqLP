from src.seqlp.visualize.load_model import DataPipeline


def test_DataPipeline():
    Data = DataPipeline(pca = False, no_sequences= 10)
    final_data = Data.sequences_array
    experiment_names = Data.init_sequencing_report["Experiment"]
    assert final_data.shape[0] == 70, "The number of sequences should be 70"
    
    