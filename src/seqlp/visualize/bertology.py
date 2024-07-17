import torch
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from Bio.PDB import PDBParser
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB.Polypeptide import PPBuilder
import matplotlib.cm as cm
import matplotlib.colors as colors
from iglabel import IMGT

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
    
    def compute_pa_f_slow(self, attentions):
        """This function is just here to showcase the logic of the algorithm better. It is not optimized and should not be used in production.

        Args:
            attentions (tensor): Returned attentions from the model given your sequence length.

        Returns:
            tensor: A 3 dimensional tensor with shape [batch_size, layers, heads]. This tensor contains the average focus for each head for the constrained positions.
        """
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

    def compute_pa_f_fast(self, attentions):
        """This algorithm is the optimized version of comoute_pa_f_slow. It uses the same logic but is optimized for speed.

        Args:
            attentions (tensor): Returned attentions from the model given your sequence length.

        Returns:
            tensor: A 3 dimensional tensor with shape [batch_size, layers, heads]. This tensor contains the average focus for each head for the constrained positions.
        """
        assert len(attentions.shape) == 5, "The attentions tensor must have shape [batch_size, layers, heads, max_seq_len, max_seq_len]"
        assert attentions.shape[3] == len(self.sequence), f"The sequence length of the attentions tensor must match the length of the sequence. You should remove the CLS and SEP token. The fourth dimension of the tensor is {attentions.shape[3]}. The sequence length is {len(self.sequence)}."

        # Convert residues list to a tensor and ensure it's on the same device as attentions
        residues_tensor = torch.tensor(self.residues, device=attentions.device)

        # Create meshgrid for indices
        seq_len = attentions.shape[3]
        i_indices, j_indices = torch.meshgrid(torch.arange(seq_len), torch.arange(seq_len), indexing='ij')
        i_indices = i_indices.to(attentions.device)
        j_indices = j_indices.to(attentions.device)

        # Apply the decision function across all indices, check for each i and j if it's in residues
        decision_values = (i_indices[..., None] == residues_tensor).any(dim=-1) | (j_indices[..., None] == residues_tensor).any(dim=-1)

        # Mask to apply where attentions are greater than zero
        mask = (attentions > 0)

        # Element-wise multiplication of decision values with attentions across all sequences, layers, and heads where mask is true
        weighted_decision = decision_values.float() * attentions * mask

        # Compute the sum of weights where attentions are positive
        weighted_sum = torch.sum(weighted_decision, dim=(-2, -1))  # sum over the last two dimensions
        sum_weights = torch.sum(attentions * mask, dim=(-2, -1))  # sum over the last two dimensions

        # Avoid division by zero
        sum_weights[sum_weights == 0] = 1

        # Calculate pa_f as the weighted mean
        pa_f = weighted_sum / sum_weights

        return pa_f
    
    def extract_sequence_from_pdb(pdb_file):
        # Create a PDB parser object
        parser = PDBParser(QUIET=True)
        
        # Parse the structure from the PDB file
        structure = parser.get_structure('structure', pdb_file)
        
        # Create a polypeptide builder object
        ppb = PPBuilder()
        
        # Extract the sequences
        sequences = []
        for pp in ppb.build_peptides(structure):
            sequences.append(pp.get_sequence())
        
        return sequences
        
import seaborn as sns



class AttentionAnalysis:
    @staticmethod
    def compute_mean_heads(heads_top:np.array):
        return np.mean(heads_top, axis = 0)
    @staticmethod
    def compute_weighted_heads(heads_top, relative_weights):
        weighted_array = relative_weights[:, np.newaxis, np.newaxis] * heads_top
        return np.sum(weighted_array, axis = 0)
    
    def calculate_top_head_one(self, matrix:torch.tensor, no_heads_average:int, attentions:torch.tensor, method = "relative_weighted") -> np.array:
        """Calculates the representative attention focus based on the number of top heads. 

        Args:
            matrix (torch.tensor): This is pa_f (return from Bertology.compute_pa_f_fast) with shape [batch_size, layers, heads]. This tensor contains the average focus for each head for the constrained positions.
            no_heads_average (int): The number of heads you would like to take the attentions from 
            attentions (torch.tensor): weights from your model for the given sequence length
            method (str, optional): _description_. Defaults to "relative_weighted".

        Returns:
            np.array: A two dimensional array which represents the attention for the given sequence
        """
        layers_head_tensor = matrix[0, :, :]
        layers_head_numpy = layers_head_tensor.cpu().numpy()
        sorted_indices = np.argsort(layers_head_numpy, axis=None)[::-1]
        top_indices_flat = sorted_indices[:no_heads_average] 
        top_indices = np.unravel_index(top_indices_flat, layers_head_numpy.shape)
        heads_top = attentions[0, top_indices[0], top_indices[1], :, :].cpu().numpy()
        weights = layers_head_numpy[top_indices[0], top_indices[1]]
        relative_weights = torch.softmax(torch.tensor(weights), dim = 0).cpu().numpy()
        assert len(heads_top.shape) == 3, "The shape of the heads_top should be 3"
        assert heads_top.shape[0] == no_heads_average, "The number of heads should be the same as the no_heads_average"
        if method == "mean":
            heads_top_one = self.compute_mean_heads(heads_top)
        elif method == "relative_weighted":
            heads_top_one = self.compute_weighted_heads(heads_top, relative_weights)
        else:
            raise ValueError("The method should be either mean or relative_weighted")

        return heads_top_one
    
    

class PlotAttention:
    def __init__(self, matrix, cmap = "Blues", figure_no = 1, ax = None):
        self.matrix = matrix
        if ax == None:
            self.create_fig(figure_no)
        else: 
            self.ax = ax
        self.cmap = cmap
        self.font_settings_title = {'fontfamily': 'serif', 'fontsize': '18', 'fontstyle': 'normal', 'fontweight': 'bold'}
        self.font_settings_normal = {'fontfamily': 'serif', 'fontsize': '14', 'fontstyle': 'normal'}

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


    
    def plot_residue_residue_all(self, sequence:str, last_residue:int, attentions,residues:list[list] = None, no_heads_average = 5, ax = None):
        """This function creates a heatmap of the mean attention weights for the top n heads given the residue settings of pa_f.

        Args:
            sequence (str): Sequence which was the input of the model
            attentions (tensor): Tensor with shape [batch_size, layers, heads, max_seq_len, max_seq_len]. This contains the attention weights of the model.
            no_heads_average (int, optional): Here you choose how many of the top attention heads for the given constraints you would like to choose to calculate the average from. Defaults to 5.
        """
        if ax == None:
            ax = self.ax
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
        sns.heatmap(heads_top_mean, cmap = self.cmap, ax = ax, cbar=None)
        if residues != None:
            colors_heatmap = ['#AEC6CF', '#77DD77', '#FFB7C5']
            for cdr, color, label in zip(residues, colors_heatmap, ['CDR1', 'CDR2', 'CDR3']):
                ax.hlines(y=-0.1, xmin=cdr[0], xmax=cdr[-1], color=color, linewidth=5)
                ax.vlines(x=last_residue, ymin=cdr[0], ymax=cdr[-1], color=color, linewidth=5)
                ax.text((cdr[0] + cdr[-1]) / 2, -3, label, color=color, ha='center', va='top')
                ax.text(last_residue + 1, (cdr[0] + cdr[-1]) / 2, label, color=color, va='center', ha='left')

            # Highlight CDR regions using axvspan and axhspan
            for cdr, color in zip(residues, colors_heatmap):
                ax.axvspan(cdr[0], cdr[-1], color=color, alpha=0.3)
                ax.axhspan(cdr[0], cdr[-1], color=color, alpha=0.3)
     #   self.ax.set_title(f"Mean Attention for top {no_heads_average}.")
        return heads_top_mean
     
    @staticmethod
    def get_labels(region_string:str, chosen_seq_length:int):
        """Creates the IMGT labels for the chosen sequence length

        Args:
            region_string (str): _description_
            chosen_seq_length (int): _description_

        Returns:
            list: A list with strings of either the IMGT labels or the position numbers
        """
        if region_string != "targetSequences":
            region = [region_string.replace("aaSeq", "")]
            label_dict, _ = IMGT([chosen_seq_length * "A"], region, save = False)
            numbers_true = list(label_dict.values())[0]
        else:
            numbers_true = [str(i) for i in list(range(1, chosen_seq_length + 1))]
        return numbers_true
     
    def create_region_specific_subplot(self, sequence:str, residues:list[list], last_residue:int, attentions, no_heads_average = 5,  title = "Average Attention over most suitable heads for the sequence", height_ratios = None, plot_imgt_labels = True):
        plt.close()
        no_regions = len(residues)
        self.fig = plt.figure(figsize=(12, 8))
        if residues is not None:
            no_cols = 3
            gs = self.fig.add_gridspec(no_cols, no_regions, width_ratios=[2, 1, 0.1], height_ratios=height_ratios, wspace=0.3, hspace=0.3)
        else:
            gs = self.fig.add_gridspec(no_cols, 3, width_ratios=[2, 1, 0.1], wspace=0.3)
        left_main = self.fig.add_subplot(gs[:, 0])
        heads_top_mean = self.plot_residue_residue_all(sequence, residues = residues, last_residue = last_residue, attentions = attentions, no_heads_average=no_heads_average, ax=left_main)
        left_main.set_xlabel("Residue position", **self.font_settings_normal)
        left_main.set_ylabel("Residue position", **self.font_settings_normal)
        left_main.set_title("Attention for the entire sequence", pad = 20, **self.font_settings_normal)
        global_min = np.min(heads_top_mean)
        global_max = np.max(heads_top_mean)
        n = 1
        avail_regions = ["CDR1", "CDR2", "CDR3"]
        if residues is not None:
            for no_region, region in enumerate(residues):
                cur_ax = self.fig.add_subplot(gs[no_region, 1])
                local_region = heads_top_mean[region[0]:region[-1], region[0]:region[-1]] 
                xtick_positions = np.arange(len(local_region)) + 0.5
                ytick_positions = np.arange(len(local_region)) + 0.5
                sequence_region = sequence[region[0]:region[-1]]
                sns.heatmap(local_region, cmap = self.cmap, ax = cur_ax, vmin=global_min, vmax=global_max, cbar = None)
                if plot_imgt_labels:
                    labels = self.get_labels(avail_regions[no_region], len(residues[no_region]) - 1)
                else:
                    labels = sequence_region
                cur_ax.set_xticks(xtick_positions)
                cur_ax.set_xticklabels(labels, ha = "center", fontsize = 7, rotation = 90)       
                cur_ax.set_yticks(ytick_positions)
                cur_ax.set_yticklabels(labels, va = "center", rotation = 0, fontsize = 7)   
                cur_ax.set_title(f"Attention for CDR{n}", **self.font_settings_normal) 
                n += 1
        cur_ax.set_xlabel("Amino acids", **self.font_settings_normal)  
        ax_colorbar = self.fig.add_subplot(gs[:, no_cols - 1])
    
        sm = cm.ScalarMappable(cmap=self.cmap, norm=colors.Normalize(vmin=global_min, vmax=global_max))
        self.fig.suptitle(t = title, **self.font_settings_title)
        cbar = plt.colorbar(sm, cax=ax_colorbar)
        
        cbar.set_label('Average Attention', **self.font_settings_normal)
        cbar.ax.yaxis.label.set_rotation(270)
        cbar.ax.yaxis.labelpad = 20 
        sm.set_array([])
        plt.tight_layout()
            
                
        
        



#comparative = collection_array[0, :, :] - collection_array[1, :, :]
#plt.subplot(1, 3, 3)  # 1 row, 2 columns, second subplot
#sns.heatmap(comparative, cmap='coolwarm')
#plt.title('Comparative')
#plt.show()


#import matplotlib

#matplotlib.use("QtAgg")
#first_array = collection_array[0, :, :]
#differences = []
#for i in range(collection_array.shape[0]):
#    diff = hellinger_distance(first_array, collection_array[i, :, :])
#    differences.append(diff)
# Create a figure and an axis
#fig, ax = plt.subplots()

# Plot each point and label it
#y_values = np.zeros_like(differences)
#df = pd.DataFrame({
 #   'Differences': differences,
  #  'Group': experiments
#})
#x_values = np.arange(len(differences))
#sns.stripplot(data=df, x=x_values, y="Differences", hue='Group', jitter=True, dodge=True, marker='o', palette='Set2')
#plt.legend(title='Experiment Groups')
#plt.title('Hellinger distance of attention distributions')
#plt.xlabel('sequence no.')  # Label the x-axis
#plt.tight_layout()
#plt.show()
# Show the plot
    


#plt.figure(figsize=(12, 5))  # Set the overall figure size

# Plot the first heatmap
#plt.subplot(1, 3, 1)  # 1 row, 2 columns, first subplot
#sns.heatmap(collection_array[0, :, :], cmap='Blues')  # 'annot=True' to annotate cells with values
#plt.title('Binding Sequence')

# Plot the second heatmap
#plt.subplot(1, 3, 2)  # 1 row, 2 columns, second subplot
#sns.heatmap(collection_array[1, :, :], cmap='Blues')
#plt.title('Non binding sequence')

#comparative = collection_array[0, :, :] - collection_array[1, :, :]
#plt.subplot(1, 3, 3)  # 1 row, 2 columns, second subplot
#sns.heatmap(comparative, cmap='coolwarm')
#plt.title('Comparative')
#plt.show()

