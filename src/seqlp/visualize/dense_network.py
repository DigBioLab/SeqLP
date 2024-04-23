from sklearn.cluster import KMeans
from load_model import LoadModel
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Assuming you have a pre-trained BERT model and tokenizer
Setup = LoadModel(model_path = r"C:\Users\nilsh\my_projects\SeqLP\tests\test_data\nanobody_model")
pca = True
def calculate_cdr_positions(row):
    # Concatenate all regions into one sequence
    full_sequence = ''.join(row)
    # Store the running length of the sequence to calculate positions
    running_length = 0
    # List to keep the position ranges of each CDR
    cdr_positions = []

    # Iterate over each region in the row
    for key in row.index:
        current_length = len(row[key])
        if 'CDR' in key:  # Check if the region is a CDR
            # Append the start and end positions to the list
            cdr_positions.append((running_length, running_length + current_length - 1))
        # Update running length after processing each region
        running_length += current_length
    
    return full_sequence, cdr_positions
# Encode your sequences
sequencing_report = pd.read_csv(r"C:\Users\nilsh\my_projects\ExpoSeq\my_experiments\max_new\sequencing_report.csv")
experiments = sequencing_report['Experiment'].tolist()
sequencing_report = sequencing_report[["aaSeqCDR1","aaSeqFR2","aaSeqCDR2","aaSeqFR3","aaSeqCDR3","aaSeqFR4"]]
sequencing_report[['full_sequence', 'CDRPositions']] = sequencing_report.apply(calculate_cdr_positions, axis=1, result_type='expand')
sequences_list = []
full_sequences = sequencing_report['full_sequence'].tolist()
for seq in full_sequences:
    inputs = Setup._get_encodings([seq])
    outputs = Setup.model(**inputs)
    last_hidden_state = outputs.last_hidden_state

    maximum_length = last_hidden_state.shape[1]

    avg_seq = np.squeeze(last_hidden_state, axis=0)

    avg_seq = last_hidden_state.mean(dim = 1) # take average for each feature from all amino acids
    sequences_list.append(avg_seq.cpu().detach().numpy()[0])
    
sequences_array = np.array(sequences_list)
assert sequences_array.shape[0] == sequencing_report.shape[0], "The number of sequences and the number of rows in the report do not match"
def do_pca(sequences_list):
    pca_components = 10
    pca = PCA(n_components=pca_components)
    pca.fit(sequences_list)
    X = pca.transform(sequences_list)

    print("Explained variance after reducing to " + str(pca_components) + " dimensions:" + str(np.sum(pca.explained_variance_ratio_).tolist()))
    return X
if pca == True:
    sequences_list = do_pca(sequences_list)
# Apply K-means clustering

### Unsupervised learning
kmeans = KMeans(n_clusters=7, random_state=0, max_iter = 300).fit(sequences_list)
cluster_labels = kmeans.labels_
sequencing_report['cluster'] = cluster_labels
sequencing_report['Experiment'] = experiments
sequencing_report.to_csv("clustered_report.csv", index = False)
# Now `cluster_labels` contains the cluster assignment for each sequence
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

y = sequencing_report['Experiment'].tolist()
y_encoded = [0 if item == "cLNTX_non-bind" else 1 for item in y]
X = sequences_list



# Assuming X is your features matrix and y is your labels vector
encoder = LabelEncoder()

# Initialize the classifier
model = model = LogisticRegression(class_weight = "balanced")

# Setup cross-validation
cv = StratifiedKFold(n_splits=5)

# Perform cross-validation
scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')

print("Cross-validated scores:", scores)
print("Average accuracy:", scores.mean())


y_encoded = encoder.fit_transform(y)

model = RandomForestClassifier(class_weight = "balanced")

# Setup cross-validation
cv = StratifiedKFold(n_splits=5)

# Perform cross-validation
scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy')

print("Cross-validated scores:", scores)
print("Average accuracy:", scores.mean())


import torch
from torch import nn
class Dense(nn.Module):
    def __init__(self,no_pc_components = 10, no_output_classes = 2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classifier = nn.Sequential(
            nn.Linear(no_pc_components, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, no_output_classes)
        
        )
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.float() 
        logits = self.classifier(x)

        return logits
    

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import numpy as np
 
# Cross-validation setup
k_folds = 5
y_encoded = [0 if item == "cLNTX_non-bind" else 1 for item in y]
y_encoded = torch.tensor(y_encoded)
kfold = KFold(n_splits=k_folds, shuffle=True)
class_counts = torch.tensor([(y_encoded == c).sum() for c in torch.unique(y_encoded)], dtype=torch.float)
class_weights = 1. / class_counts
class_weights = class_weights / class_weights.sum()
weighted_loss = nn.CrossEntropyLoss(weight=class_weights)

# Training loop
def train_loop(dataloader, model, optimizer, weighted_loss):
    model.train()  # Set model to training mode
    for X, y in dataloader:
        pred = model(X)
        loss = weighted_loss(pred, y)  # Use the weighted loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Validation loop
def validate_loop(dataloader, model, weighted_loss):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = weighted_loss(pred, y)  # Use the weighted loss function
            total_loss += loss.item()
            total_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    accuracy = total_correct / size
    print(f"Validation: Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

batch_size = 15
X = torch.tensor(X, dtype=torch.float32) 
for fold, (train_ids, test_ids) in enumerate(kfold.split(X.numpy())):  # Ensure correct handling of indices
    print(f"FOLD {fold}")
    print("-------------------------------")

    # Sample elements randomly from a given list of indices, no replacement.
    X_train_fold, y_train_fold = X[train_ids], y_encoded[train_ids]
    X_test_fold, y_test_fold = X[test_ids], y_encoded[test_ids]

    trainloader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=batch_size)
    testloader = DataLoader(TensorDataset(X_test_fold, y_test_fold), batch_size=batch_size)

    # Initialize model and optimizer for each fold
    model = Dense()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Weighted loss
    class_counts = torch.tensor([(y_train_fold == c).sum() for c in torch.unique(y_train_fold)], dtype=torch.float32)
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    weighted_loss = nn.CrossEntropyLoss(weight=class_weights)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        train_loop(trainloader, model, optimizer, weighted_loss)
        validate_loop(testloader, model, weighted_loss)
        print("---------------------------------")


    



