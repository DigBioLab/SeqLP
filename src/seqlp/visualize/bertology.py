import torch
import matplotlib.pyplot as plt


import numpy as np

# property function f is often symmetric: which means that i,j and j,i return 1 or 0, respectively. 
# The asymmetric case when i,j = 1 and j,i = 0 would happen if your attention is direction dependent. Direction should not be important for protein structures

# we will start with symmetric properties - so you parse only one list of residues
import types
class Bertology:
    def __init__(self, residues:list = None, sequence:str = None, decision_function = "binding_site"):
        """_summary_

        Args:
            residues (list): This is a list of indexes of residues in the sequence.
            decision_function: This object can either be a self defined function or a string which is a method implemented in this class.
                                Available methods are: binding_site, hydrobphob
        """
        self.residues = residues
        if isinstance(decision_function, types.FunctionType):
            self.decision_func = decision_function
        else:
            assert hasattr(self, f"_f_{decision_function}"), f"The function {decision_function} is not implemented. Please choose one of the implemented functions or provide a custom function."
            self.decision_func = getattr(self, f"_f_{decision_function}")
        self.sequence = sequence
        
    def _f_binding_site(self, i, j):
        """Symmetric function. It returns 1 if i or j is in residues, otherwise 0. can be used if only the position of the residues is of interest (interesting for cdr3)

        Args:
            i (_type_): x value in specific attention head
            j (_type_): y value in specific attention head

        Returns:
            _type_: 
        """
        if i in self.residues or j in self.residues:
            return 1
        else:
            return 0


    # a weighted decision function could be interesting which depends on multiple parameters and which sum of weights is 1
    
    def _f_hydrophob(self, i, j, ):
        kyte_doolittle = {
            'R': -4.5, 'K': -3.9, 'D': -3.5, 'E': -3.5, 'N': -3.5, 'Q': -3.5,
            'H': -3.2, 'S': -0.8, 'T': -0.7, 'G': -0.4, 'A': 1.8, 'M': 1.9,
            'C': 2.5, 'Y': -1.3, 'W': -0.9, 'P': -1.6, 'V': 4.2, 'I': 4.5,
            'L': 3.8, 'F': 2.8, "_": -4.5, "*": -4.5
        }

        min_value = min(kyte_doolittle.values())
        max_value = max(kyte_doolittle.values())

        normalized_kyte_doolittle = {key: (value - min_value) / (max_value - min_value) for key, value in kyte_doolittle.items()}
        # doesnt work well, probably because hydrobhobicity is a very universal characteristic which may appear in each head
        aa_i = self.sequence[i]
        aa_j = self.sequence[j]
        hydrophobicity_vector_mean = normalized_kyte_doolittle[aa_i] * normalized_kyte_doolittle[aa_j]
        return hydrophobicity_vector_mean
    

    # should be reprogrammed with matrix multiplication, because it is very slow currently
    def compute_pa_f(self, attentions):
        assert len(attentions.shape) == 5, "The attentions tensor must have shape [batch_size, layers, heads, max_seq_len, max_seq_len]"
        assert attentions.shape[3] == len(self.sequence), f"The sequence length of the attentions tensor must match the length of the sequence. You should remove the CLS and SEP token. The fourth dimension of the tensor is {attentions.shape[3]}. The sequence length is {len(self.sequence)}."
        numerator = torch.zeros((attentions.shape[0], attentions.shape[1], attentions.shape[2]), device=attentions.device) # shape [batch_size, layers, heads] holds the sum of each sequence, layer and head
        denominator = torch.zeros_like(numerator)
        for sequence in range(attentions.shape[0]):
            for layer in range(attentions.shape[1]):
                # Iterate through each head in the current layer
                for head in range(attentions.shape[2]):
                    # Iterate through the rows of the attention matrix for the current head
                    for i in range(0, attentions.shape[3]):
                        # Iterate through the columns of the attention matrix for the current token
                        for j in range(0,attentions.shape[4]):
                            # Check if the attention weight is greater than zero
                            alpha_ij = attentions[sequence, layer, head, i, j]
                            if alpha_ij > 0:
                                # Apply function f to the indices and multiply by the attention weight
                                numerator[sequence, layer, head] += self.decision_func(i, j) * alpha_ij
                                denominator[sequence, layer, head] += alpha_ij
        pa_f = numerator / denominator # should be the mean of the attention weights per head
        assert pa_f.shape[0] == attentions.shape[0], "The batch size of the pa_f tensor should match the attentions tensor"
        assert pa_f.shape[1] == attentions.shape[1], "The number of layers of the pa_f tensor should match the attentions tensor"
        assert pa_f.shape[2] == attentions.shape[2], "The number of heads of the pa_f tensor should match the attentions tensor"
        return pa_f
        
import seaborn as sns

class PlotAttention:
    def __init__(self, matrix, cmap = "Blues", figure_no = 1, ax = None):
        self.matrix = matrix
        if ax == None:
            self.create_fig(figure_no)
        else: 
            self.ax = ax
        self.cmap = cmap

    def create_fig(self, figure_no):
        self.fig = plt.figure(figure_no)
        self.ax = self.fig.gca()
        
    def plot_mean_head(self, sequence:str, residues:list):
        assert len(self.matrix.shape) == 3, "The input matrix should three dimensions. the first one is the batch size."
        layers_head_tensor = self.matrix[0, :, :]
        layers_head_numpy = layers_head_tensor.cpu().numpy()
        sns.heatmap(layers_head_numpy, ax = self.ax, cmap = self.cmap) # upper row represents the first layer
        self.ax.set_xlabel("Heads")
        self.ax.set_ylabel("Layers")
        self.ax.set_title(f"Mean Attention per Head for {sequence} and residues {residues}")


    def plot_residue_residue(self, sequence:str, attentions, no_heads_average = 5):
        """This function creates a heatmap of the mean attention weights for the top n heads given the residue settings of pa_f.

        Args:
            sequence (str): Sequence which was the input of the model
            attentions (tensor): Tensor with shape [batch_size, layers, heads, max_seq_len, max_seq_len]. This contains the attention weights of the model.
            no_heads_average (int, optional): Here you choose how many of the top attention heads for the given constraints you would like to choose to calculate the average from. Defaults to 5.
        """
        assert len(self.matrix.shape) == 3, "The input matrix should three dimensions. the first one is the batch size."
        assert type(sequence) == str, "The sequence should be a string"
        layers_head_tensor = self.matrix[0, :, :]
        layers_head_numpy = layers_head_tensor.cpu().numpy()
        sorted_indices = np.argsort(layers_head_numpy, axis=None)[::-1]
        top_indices_flat = sorted_indices[:no_heads_average] 
        top_indices = np.unravel_index(top_indices_flat, layers_head_numpy.shape)
        heads_top = attentions[0, top_indices[0], top_indices[1], :, :].cpu().numpy()
        assert len(heads_top.shape) == 3, "The shape of the heads_top should be 3"
        assert heads_top.shape[0] == no_heads_average, "The number of heads should be the same as the no_heads_average"
        heads_top_mean = np.mean(heads_top, axis = 0) # you average over first dimension which is the number of heads
        sns.heatmap(heads_top_mean, cmap = "Blues", ax = self.ax)
        self.ax.set_xticks(np.arange(len(sequence)) + 0.5)  # Centering the labels
        self.ax.set_xticklabels(list(sequence), rotation=90, ha='right')  # Setting labels to letters
        self.ax.set_yticks(np.arange(len(sequence)) + 0.5)  # Centering the labels
        self.ax.set_yticklabels(list(sequence))  # Setting labels to letters
        self.ax.set_title(f"Mean Attention for top {no_heads_average}.")





