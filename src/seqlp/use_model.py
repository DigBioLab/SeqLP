from seqlp.visualize.load_model import LoadModel
from seqlp.visualize.tidy_protbert_embedding import TransformerBased
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import scipy.stats as stats

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score
)
import torch.nn as nn
class AnalyseModel:
    def __init__(self, model_string:str, model = None, tokenizer = None, load_masked_lm = False):
        """_summary_

        Args:
            model_string (str): Can be a model from huggingface or a path which leads to a model which was pretrained.
        """
        self.ModelSets = LoadModel(model_string, model, tokenizer, load_masked_lm=load_masked_lm)
        self.font_settings_title = {'fontfamily': 'serif', 'fontsize': '18', 'fontstyle': 'normal', 'fontweight': 'bold'}
        self.font_settings_normal = {'fontfamily': 'serif', 'fontsize': '14', 'fontstyle': 'normal'}
        
    def make_figure(self, backend = "Qt5Agg"):
        plt.close('all')
        plt.switch_backend(backend)
        self.fig = plt.figure(1, figsize = (12, 10))
        self.ax = self.fig.gca()

        
    def update_plot(self, figure_style = "seaborn-v0_8-colorblind"):
        self.ax = self.fig.gca()
        plt.style.use(figure_style)
        self.ax.spines['right'].set_visible(False) 
        self.ax.spines['top'].set_visible(False)
        plt.tight_layout()
        
        
        
    def _embed_sequences(self, sequences) -> np.array:
        if not type(sequences) == list:
            raise ValueError("Please provide the sequences in a list format")
        assert type(sequences[0]) == str, "Please provide the sequences in a list of strings"
        return self.ModelSets._get_embeddings_parallel(sequences)
        
    @staticmethod
    def _get_scalarmappable(min, max, prefered_cmap):
        sm = plt.cm.ScalarMappable(
                cmap=prefered_cmap,
                norm=plt.Normalize(vmin=min, vmax=max),
            )
        norm = plt.Normalize(vmin=min, vmax=max)
        return sm, norm
    
    
    def add_colorbar(self, label, sm, cmap):
        colorbar_settings = { 'orientation': 'horizontal', 'spacing': 'proportional', 'extend': 'neither'}
        colorbar_settings["cmap"] = cmap
        #    del colorbar_settings["spacing"]
        fig = self.ax.get_figure()
        fig.colorbar(
            sm,
            ax=self.ax,
            label=label,
            **colorbar_settings,
        )

    def add_legend(self, title, handling = True):
        from matplotlib.font_manager import FontProperties
        legend_settings = {'loc': 'center left','facecolor': 'black',  'bbox_to_anchor': (1, 0.5), 'ncols': 1, 'fontsize': 16, 'frameon': True, 'framealpha': 1, 'facecolor': 'white', 'mode': None, 'title_fontsize': 'large', 'title_fontsize': 'large'}
        legend_settings["prop"] = FontProperties(size=14)
        lgnd = self.ax.legend(**legend_settings)
        if handling:
            for legend_handle in lgnd.legendHandles:
                legend_handle._sizes = [25]
                legend_handle.set_color("black")
                legend_handle.set_facecolor("black")
       # lgnd.set_title(title)
        
        
    @staticmethod
    def do_pca(sequences_list, explained_variance_threshold):
        """
        Perform PCA to reduce the dimensionality based on explained variance threshold.
        
        Args:
        sequences_list: List or array-like, shape (n_samples, n_features)
            High-dimensional data needing dimensionality reduction.
        explained_variance_threshold: float
            The fraction of explained variance (e.g., 0.95 for 95% of variance to be retained).

        Returns:
        X: array, shape (n_samples, n_components)
            The matrix of principal components with reduced dimensionality.
        """
        # Initialize PCA without specifying the number of components
        pca = PCA()
        pca.fit(sequences_list)

        # Calculate the cumulative explained variance and determine the optimal number of components
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.where(cumulative_variance >= explained_variance_threshold)[0][0] + 1

        # Run PCA again with the optimal number of components
        pca = PCA(n_components=n_components)
        pca.fit(sequences_list)
        X = pca.transform(sequences_list)
        
        # Optionally print the explained variance
        print(f"Explained variance after reducing to {n_components} dimensions: {cumulative_variance[n_components-1]:.2f}")

        return X, n_components
    
    def pca_driven(self, n_components, sequences_list):
        pca = PCA(n_components=n_components)
        pca.fit(sequences_list)
        X = pca.transform(sequences_list)
        return X
    
    def pca_component_driven(self, n_components, sequences_list):
        pca = PCA(n_components=n_components)
        pca.fit(sequences_list)
        X = pca.transform(sequences_list)
        return X
    
    def add_title(self, title):
        self.ax.set_title(title, pad=20, **self.font_settings_title)
    
    def embed_cluster_sequences(self, sequences:list, labels:list = None, color_gradient:list = None, 
                                explained_variance_threshold = 0.9, n_neighbors = 15, min_dist = 0.1, alpha = 0.5, cmap = "viridis", legend_header = "Targets",title = ""):
        """Creates a plot with the embeddings of the sequences as dots and if labels are provided, depending on the label n markers are introduced. Furthermore the dots are colored according to color gradient.

        Args:
            sequences (list): Amino acid sequences without spaces between the amino acid
            labels (list, optional): List of labels for the sequences. Length must be equal to sequences. Defaults to None.
            color_gradient (list, optional): _description_. Defaults to None.
            explained_variance_threshold (float, optional): _description_. Defaults to 0.9.
            n_neighbors (int, optional): _description_. Defaults to 15.
            min_dist (float, optional): _description_. Defaults to 0.1.
            alpha (float, optional): _description_. Defaults to 0.5.
            cmap (str, optional): _description_. Defaults to "viridis".
            legend_header (str, optional): _description_. Defaults to "Targets".
            title (str, optional): _description_. Defaults to "".
        """
        if labels is not None:
            assert len(labels) == len(sequences), "The length of the labels should be equal to the length of the sequences"
        if color_gradient is not None:
            assert len(color_gradient) == len(sequences), "The length of the color_gradient should be equal to the length of the sequences"
        assert type(sequences) == list, "Please provide the sequences in a list format"
        assert type(sequences[0]) == str, "Please provide the sequences in a list of strings"
        embedding_array = self._embed_sequences(sequences)
        X, _ = self.do_pca(embedding_array, explained_variance_threshold)
        reduced_X , _ = TransformerBased.do_umap(X, n_neighbors = n_neighbors, min_dist = min_dist, metric= "euclidean") # UMAP_1, UMAP_2
        self.make_figure()
        if labels is not None:
            reduced_X["labels"] = labels
        if color_gradient is not None:
            reduced_X["color_gradient"] = color_gradient
        
        markers = ["o", "s", "+", "X", "^", "*", "p", "H", "_", "D", "v", "|"]
        if labels != None:
            unique_labels = reduced_X["labels"].unique()
            for index, unique_label in enumerate(unique_labels):
                if labels is not None:
                    localX = reduced_X[reduced_X["labels"] == unique_label]
                if color_gradient is not None:
                    color_gradient = localX["color_gradient"].values
                if labels != None:
                    self.ax.scatter(localX["UMAP_1"], localX["UMAP_2"],
                                    marker = markers[index], c = color_gradient, label = unique_label, alpha = alpha, cmap = cmap)
        else:
            self.ax.scatter(reduced_X["UMAP_1"], reduced_X["UMAP_2"], c = color_gradient, cmap = cmap, alpha = alpha)
        if color_gradient is not None:
            sm, norm = self._get_scalarmappable(reduced_X["color_gradient"].min(), reduced_X["color_gradient"].max(), cmap)
            self.add_colorbar("Binding Affinity", sm, cmap)
        if labels is not None:
            self.add_legend(legend_header)
        self.ax.set_xlabel("UMAP_1", **self.font_settings_normal)
        self.ax.set_ylabel("UMAP_2", **self.font_settings_normal)
        self.add_title(title)
        self.update_plot()
        
    def no_components_for_sequences(self, sequences:list, explained_variance_threshold = 0.9):
        embedding_array = self._embed_sequences(sequences)
        X, no_components = self.do_pca(embedding_array, explained_variance_threshold)
        return no_components
        
    def embed_cluster_label(self, sequences:list, labels:list = None, embeddings = None,
                                explained_variance_threshold = 0.9,  n_components = None, n_neighbors = 15, min_dist = 0.1, alpha = 0.75, size_points = 15, cmap = "viridis", legend_header = "Targets",title = "", color_list = None):
        """_summary_

        Args:
            sequences (list): Amino acid sequences without spaces between the amino acid
            labels (list, optional): List of labels for the sequences. Length must be equal to sequences. Defaults to None.
            embeddings (np.array, optional): If you provide those the algorithm will skip the use of the model and continues with the plot.
            explained_variance_threshold (float, optional): If you provide this the algorithm will look for n pc components to reach this explained variance. Defaults to 0.9.
            n_components (_type_, optional): If you provide this the algorithm will use that many components to pipe into UMAP. Defaults to None.
            n_neighbors (int, optional): Number of neighbors for UMAP. Defaults to 15.
            min_dist (float, optional): _description_. Defaults to 0.1.
            alpha (float, optional): _description_. Defaults to 0.5.
            size_points (int, optional): _description_. Defaults to 15.
            cmap (str, optional): _description_. Defaults to "viridis".
            legend_header (str, optional): _description_. Defaults to "Targets".
            title (str, optional): _description_. Defaults to "".
            color_list (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if labels is not None:
            assert len(labels) == len(sequences), "The length of the labels should be equal to the length of the sequences"

        assert type(sequences) == list, "Please provide the sequences in a list format"
        assert type(sequences[0]) == str, "Please provide the sequences in a list of strings"
        if embeddings is not None:
            embedding_array = embeddings
        else:
            embedding_array = self._embed_sequences(sequences)
        if n_components is not None:
            X:np.array = self.pca_driven(n_components, embedding_array)
        else:
            X, _ = self.do_pca(embedding_array, explained_variance_threshold)
        reduced_X , _ = TransformerBased.do_umap(X, n_neighbors = n_neighbors, min_dist = min_dist, metric= "euclidean") # UMAP_1, UMAP_2
        self.make_figure()
        if labels is not None:
            reduced_X["labels"] = labels
            reduced_X["labels_factorized"], unique_codes = pd.factorize(reduced_X["labels"])
        reduced_X["sequences"] = sequences
        unique_labels = reduced_X["labels"].unique()
        cmap_string = cmap
        cmap = plt.get_cmap(cmap)
        if color_list is None:
            color_list = [cmap(i / (len(unique_labels) - 1)) for i in range(len(unique_labels))]
        
        color_list = [(r, g, b, alpha) for r, g, b, _ in color_list]
        print(color_list)
        for index, unique_label in enumerate(unique_labels):
            
            localX = reduced_X[reduced_X["labels"] == unique_label]
            self.ax.scatter(localX["UMAP_1"], localX["UMAP_2"], s = size_points, c = color_list[index], label = unique_label, alpha = alpha, cmap = cmap_string)
        if labels is not None:
            self.add_legend(legend_header, handling = False)
        self.ax.set_xlabel("UMAP_1", **self.font_settings_normal)
        self.ax.set_ylabel("UMAP_2", **self.font_settings_normal)
        self.add_title(title)
        self.update_plot()
        return reduced_X
    
    def save_in_plots(self, enter_filename):
        plt.savefig(
            fname=os.path.join( enter_filename + f".png"),
            dpi=300,
            format="png",
            bbox_inches="tight",
        )
        


    
    def feature_correlation_to_pcs(self, n_components, sequences, features:list[np.array]):
        sequences_list = self._embed_sequences(sequences)
        X = self.pca_component_driven(n_components, sequences_list)
        all_features = X
        for feature in features:
            if len(feature.shape) == 1:
                feature = feature[:, np.newaxis]
            scaler = StandardScaler()
            standard_scaled_data = scaler.fit_transform(feature)
            all_features = np.hstack((all_features, standard_scaled_data))
        correlation_matrix = np.corrcoef(all_features.T)
            
        self.make_figure()
        self.ax.imshow(correlation_matrix[:, :], cmap='bwr', interpolation='none')
     #   plt.colorbar()
        plt.title('Correlation between Principal Components and External Features')
        plt.show()
        
        
    def calculate_perplexity(self, sequences:list, pad_token_id = 1):
        """Use this method to calculate the perplexity of your sequences. 

        Args:
            sequences (list): list of sequences. If they are shorter than max_length they will be padded.
            pad_token_id (int, optional): The pad token id of the model you are using. You must look at the vocab for the tokenizer to get that. Defaults to 1.

        Returns:
            _type_: _description_
        """
        ## calculation according to https://huggingface.co/docs/transformers/perplexity but without striding as this is not needed here
        encoded_input = self.ModelSets._get_encodings(sequences)
        encodings = encoded_input
        labels = encodings["input_ids"].clone()
        if "token_type_ids" in encodings:
            del encodings["token_type_ids"]
        with torch.no_grad():
            outputs = self.ModelSets.model(**encodings, labels=labels)
        metrics=  self.compute_metrics(outputs.logits, labels, pad_token_id)
        return metrics
    
    @staticmethod
    def compute_metrics(logits, labels, pad_token_id):
        """
        Calculate various metrics, ignoring padding tokens, first token (CLS), and last token (SEP). 
        """
        logits = logits.permute(0, 2, 1)  # Reformat logits to [batch, num_classes, sequence_length]
        probabilities = torch.softmax(logits, dim=1)  # Apply softmax over num_classes
        predictions = torch.argmax(probabilities, dim=1)

        # General mask for non-padding tokens
        mask = (labels != pad_token_id)
        # Create additional masks to remove the first and last tokens
        first_last_mask = torch.ones_like(labels, dtype=torch.bool)
        first_last_mask[:, 0] = 0  # Remove first token
        first_last_mask[torch.arange(first_last_mask.shape[0]), labels.argmax(dim=1)] = 0  # Remove last token assuming SEP is at the end

        # Combine masks
        final_mask = mask & first_last_mask

        # Flatten labels and logits, apply final combined mask
        labels_flat = labels.reshape(-1)[final_mask.reshape(-1)]
        predictions_flat = predictions.reshape(-1)[final_mask.reshape(-1)]
        logits_flat = logits.permute(0, 2, 1).reshape(-1, logits.size(1))[final_mask.reshape(-1)]

        # Apply the mask to logits and labels
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits_flat, labels_flat)  # Compute the loss only on non-padded and non-special tokens

        return {
            "f1": f1_score(labels_flat.cpu().numpy(), predictions_flat.cpu().numpy(), average='macro'),
            "perplexity": torch.exp(loss).item()  # Calculate perplexity from the loss
        }


    @staticmethod
    def paired_t_test(model_A_accuracies, model_B_accuracies):
        differences = [a - b for a, b in zip(model_A_accuracies, model_B_accuracies)]

        t_statistic, p_value = stats.ttest_rel(model_A_accuracies, model_B_accuracies)

        print(f"T-statistic: {t_statistic}, P-value: {p_value}")

        if p_value < 0.05:
            print("The performance difference is statistically significant.")
        else:
            print("The performance difference is not statistically significant.")