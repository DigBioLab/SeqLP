import re
import torch
from transformers import AutoTokenizer, EsmModel, AutoModel, EsmForMaskedLM
import torch.nn.functional as F
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np 


class LoadModel:
    def __init__(self, model_path:str, model = None, tokenizer = None, load_masked_lm = False):
        if model == None and tokenizer == None:
            self.load_model(model_path, load_masked_lm)
        else:
            self.model = model
            self.tokenizer = tokenizer
    
    def load_model(self, model_path:str, load_masked_lm:bool):
    #    assert os.path.isdir(model_path), "Your model path is not a directory. Please provide a directory with the model files. The files must be config.json, model.safetensors, training_args.bin"
      #  assert os.path.isfile(os.path.join(model_path, "model.safetensors")), "Your model path does not contain a model.safetensors file. Please provide a directory with the model files. The files must be config.json, model.safetensors, training_args.bin"
        if load_masked_lm == True:
            self.model = EsmForMaskedLM.from_pretrained(model_path)
        else:
            self.model = AutoModel.from_pretrained(model_path)

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    
    def _get_encodings(self, sequence):
        assert type(sequence) == list, "The sequence must be a list of strings"
        assert len(sequence) > 0, "You need to provide at least one sequence"
        assert type(sequence[0]) == str, "The sequence must be a string"
        sequences = [" ".join(list(re.sub(r"[UZOB*_]", "X", seq))) for seq in sequence if not " " in seq]
        if len(sequences) == 0:
            sequences = sequence # case when there is already a space between all letters
        encoded_input = self.tokenizer(sequences, return_tensors='pt', padding=True)
        return encoded_input
    
    def _get_embeddings(self, full_sequences):
        sequences_list = []
        for seq in full_sequences:
            inputs = self._get_encodings([seq])
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state

            maximum_length = last_hidden_state.shape[1]

            avg_seq = np.squeeze(last_hidden_state, axis=0)

            avg_seq = last_hidden_state.mean(dim = 1) # take average for each feature from all amino acids
            
            sequences_list.append(avg_seq.cpu().detach().numpy()[0])
            
        sequences_array = np.array(sequences_list)
        return sequences_array
    
    
    def _get_embeddings_parallel(self, full_sequences, batch_size=32):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)  # Move model to appropriate device

        # Prepare to collect batch results
        all_avg_seqs = []

        # Process sequences in batches
        for i in range(0, len(full_sequences), batch_size):
            batch_sequences = full_sequences[i:i + batch_size]
            inputs = self._get_encodings(batch_sequences)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device

            with torch.no_grad():  # Disable gradient computation for inference
                outputs = self.model(**inputs)
                last_hidden_state = outputs.last_hidden_state
                avg_seq = last_hidden_state.mean(dim=1)
                all_avg_seqs.append(avg_seq.cpu().detach().numpy())  # Move results back to CPU and convert to numpy

        # Concatenate all batch results into a single numpy array
        sequences_array = np.concatenate(all_avg_seqs, axis=0)
        return sequences_array
        
    def get_attention(self, sequence:list):
        """Returns attentions of the loaded model and removes the CLS and SEP tokens from those. It assumes that these are the first and last tokens in the sequence, respectively.

        Args:
            sequence (list): A list of sequences

        Returns:
            _type_: _description_
        """
        encoded_input = self._get_encodings(sequence)
        self.model.eval()
        tokens = encoded_input['input_ids']
        with torch.no_grad():
            output = self.model(**encoded_input, output_attentions=True)
        attentions = torch.stack(output['attentions'], dim=1)
                
        # Remove [CLS] and [SEP] token attentions
        # Assuming [CLS] at index 0 and [SEP] at the last index of each sequence
        seq_len = attentions.shape[3]  # Assuming all sequences are padded to the same length
        # Remove the first and last token (typically [CLS] and [SEP])
        attentions = attentions[:, :, :, 1:seq_len-1, 1:seq_len-1]
        return attentions
    
    
    def _get_perplexity(self, sequence: str) -> float:
        # Encode the input sequence
        encoded_input = self._get_encodings([sequence])
        input_ids = encoded_input['input_ids']

        # Ensure input_ids has a batch dimension
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Adds a batch dimension
        
        # Create a mask for random tokens to be predicted (15% probability)
        mask = torch.rand(input_ids.shape) < 0.15
        masked_indices = mask.nonzero(as_tuple=True)
        
        # Check if any token is masked, and handle the case where no tokens are masked
        if masked_indices[0].numel() == 0:
            return float('inf')  # Return infinity if no token is masked, as perplexity can't be calculated

        # Store original tokens at masked positions
        original_tokens = input_ids[masked_indices].clone()

        # Replace masked token ids with the mask token id
        input_ids[masked_indices] = self.tokenizer.mask_token_id
        
        # Calculate the model output without calculating gradients
        with torch.no_grad():
            outputs = self.model(input_ids)
            last_hidden_state = outputs.last_hidden_state
            # Project last hidden state to vocabulary space using the embedding matrix
            logits = self.model.embeddings.word_embeddings.weight.matmul(last_hidden_state.transpose(1, 2)).transpose(1, 2)

        # Calculate loss manually using cross-entropy at the positions of the masked tokens
        # Only calculate loss for masked positions, reshaping as necessary for cross-entropy
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), original_tokens.view(-1), reduction='sum')
        
        # Calculate and return perplexity
        perplexity = torch.exp(loss / masked_indices[0].numel())
        return perplexity
        
    

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
    def calculate_cdr_positions(row) -> (str, tuple):
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


class DataPipeline:
    def __init__(self, model = r"C:\Users\nilsh\my_projects\ExpoSeq\models\nanobody_model", pca = True, path_seq_report = r"C:\Users\nilsh\my_projects\ExpoSeq\my_experiments\max_new\sequencing_report.csv", pca_components = 10, no_sequences =10, choose_labels = None) -> None:
        if model != None:
            self.Setup = LoadModel(model_path = model)
        else:
            self.Setup = None
        pca = pca
        self.init_sequencing_report = self._read_csv(path_seq_report, no_sequences, choose_labels)
        self.full_sequences, experiments = self.wrangle_report(self.init_sequencing_report)
        if self.Setup != None:
            self.sequences_array = self._get_encodings(self.full_sequences)
            if pca == True:
                self.X = self.do_pca(self.sequences_array, pca_components)
            else:
                self.X = self.sequences_array
                
    def _read_csv(self, path_seq_report, no_head = 100, choose_labels = None):
        csv = pd.read_csv(path_seq_report)  
        if "Experiment" in csv.columns:
            csv = csv.groupby("Experiment").head(no_head)
        else:
            csv = csv.head(100)
        if choose_labels is not None:
            csv = csv[csv["Experiment"].isin(choose_labels)]
        return csv
        
    @staticmethod
    def wrangle_report(sequencing_report):
        experiments = sequencing_report['Experiment'].tolist()
        sequencing_report = sequencing_report[["aaSeqCDR1","aaSeqFR2","aaSeqCDR2","aaSeqFR3","aaSeqCDR3","aaSeqFR4"]]
        sequencing_report[['full_sequence', 'CDRPositions']] = sequencing_report.apply(ExtractData.calculate_cdr_positions, axis=1, result_type='expand')
        full_sequences = sequencing_report['full_sequence'].tolist()
        return full_sequences, experiments
    
    def _get_encodings(self, full_sequences):
        sequences_list = []
        for seq in full_sequences:
            inputs = self.Setup._get_encodings([seq])
            outputs = self.Setup.model(**inputs)
            last_hidden_state = outputs.last_hidden_state

            maximum_length = last_hidden_state.shape[1]

            avg_seq = np.squeeze(last_hidden_state, axis=0)

            avg_seq = last_hidden_state.mean(dim = 1) # take average for each feature from all amino acids
            sequences_list.append(avg_seq.cpu().detach().numpy()[0])
            
        sequences_array = np.array(sequences_list)
        return sequences_array
    
    @staticmethod
    def do_pca(sequences_list, pca_components):
        pca = PCA(n_components=pca_components)
        pca.fit(sequences_list)
        X = pca.transform(sequences_list)

        print("Explained variance after reducing to " + str(pca_components) + " dimensions:" + str(np.sum(pca.explained_variance_ratio_).tolist()))
        return X
    
    


    
