    
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

    
class UnsupervisedML:
    def __init__(self, X) -> None:
        self.X = X
        
    def cluster_with_hdbscan(self, eps = 0.5, min_pts = 10):
        """_summary_

        Args:
            X (np.array): n values x n features
            eps (float, optional): Maximum distance between two points to still form one cluster. Defaults to 3.
            min_pts (int, optional): fewest number of points required to form a cluster. Defaults to 4.

        Returns:
            _type_: _description_
        """
        get_clusters = DBSCAN(eps = eps, min_samples = min_pts).fit_predict(self.X)
        return get_clusters
    
    def kmeans_clustering(self, n_clusters = 2):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter = 300).fit(self.X)
        cluster_labels = kmeans.labels_
        return cluster_labels
        
        
    def agglomerative_clustering(self, n_clusters = 4):
        ahc = AgglomerativeClustering(n_clusters = n_clusters,
                             metric = 'euclidean', 
                             linkage = 'ward')

        y_ahc = ahc.fit_predict(self.X)
        return y_ahc
#

from supervised_ml import DataPipeline
Data = DataPipeline(no_sequences = 100)

y_ahc = UnsupervisedML(Data.X).agglomerative_clustering(n_clusters = 7)
report = Data.init_sequencing_report
report["cluster"] = y_ahc
grouped = report.groupby(['Experiment', 'cluster']).size().unstack(fill_value=0)
report['dominant_cluster'] = report['Experiment'].map(grouped.idxmax(axis=1))
most_common_experiments = report.groupby('dominant_cluster')['Experiment'].agg(lambda x: x.mode()[0])
cluster_to_experiment = most_common_experiments.to_dict()

report['predicted_experiment'] = report['dominant_cluster'].map(cluster_to_experiment)

# Check if the predicted experiment matches the actual experiment
report['is_correct'] = (report['predicted_experiment'] == report['Experiment'])

# Calculate the accuracy for each experiment
accuracy_per_experiment = report.groupby('Experiment')['is_correct'].mean()
print(accuracy_per_experiment)
#sequencing_report['cluster'] = cluster_labels
#sequencing_report['Experiment'] = experiments