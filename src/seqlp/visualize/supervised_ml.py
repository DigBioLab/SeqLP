from sklearn.cluster import KMeans
from seqlp.visualize.load_model import LoadModel, DataPipeline
import pandas as pd
import numpy as np

# Assuming you have a pre-trained BERT model and tokenizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import torch.nn.functional as F


    
#Data = DataPipeline(pca = False, no_sequences= 1000000)
#final_data = Data.sequences_array
#experiment_names = Data.init_sequencing_report["Experiment"]
#np.savetxt("labels.txt", experiment_names, fmt='%s')
#np.savetxt("embeddings.npy", final_data)


class Dense(nn.Module):
    def __init__(self, layer_sizes, no_pc_components=10, no_output_classes=2, chosen_activation_func="GELU"):
        super(Dense, self).__init__()
        
        sizes = [no_pc_components] + layer_sizes + [no_output_classes]
        activation_func = getattr(nn, chosen_activation_func)()
        
        layers = []
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(sizes[i-1], sizes[i]))
            if i < len(sizes) - 1:  # Avoid adding activation function in the output layer
                layers.append(nn.BatchNorm1d(sizes[i]))
                layers.append(activation_func)
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float() 
        out = self.classifier(x)
        return out





class TrainNetwork:
    
    def train_loop(self, dataloader, model, optimizer, loss_fn):
        model.train()  # Set model to training mode
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)  # Use the weighted loss function

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation loop
    def validate_loop(self, dataloader, model, loss_fn):
        model.eval()  # Set model to evaluation mode
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                loss = loss_fn(pred, y)  # Use the weighted loss function
                total_loss += loss.item()
                total_correct += (torch.argmax(pred, dim = 1) == y).type(torch.float).sum().item()

        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        accuracy = total_correct / size
        print(f"Validation: Avg loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    def iterate_through_epochs(self, model, trainloader, testloader, optimizer, loss_fn, num_epochs = 5):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}")
            self.train_loop(trainloader, model, optimizer, loss_fn)
            self.validate_loop(testloader, model, loss_fn)
            print("---------------------------------")
            
            
class SetupTrain:
    def __init__(self, X, y_encoded, model, train_ids, test_ids, batch_size = 10, learning_rate = 0.00005) -> None:
        self.batch_size = batch_size
        self.X = X
        self.y_encoded = y_encoded
        self.trainloader, self.testloader = self.create_dataset(X, y_encoded, train_ids, test_ids)
        self.loss_fn = self.setup_loss_fn(y_encoded)
        self.optimizer = self._get_optimizer(model, learning_rate)
        
    def create_dataset(self,X, y_encoded, train_ids, test_ids):
        assert type(X) == torch.Tensor, "X should be a torch tensor"
        assert type(y_encoded) == torch.Tensor, "y_encoded should be a torch tensor"
        
        X_train_fold, y_train_fold = X[train_ids], y_encoded[train_ids]
        X_test_fold, y_test_fold = X[test_ids], y_encoded[test_ids]
        trainloader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=self.batch_size)
        testloader = DataLoader(TensorDataset(X_test_fold, y_test_fold), batch_size=self.batch_size)
        return trainloader, testloader
    
    @staticmethod
    def _get_optimizer(model, learning_rate):
        return optim.Adam(model.parameters(), lr= learning_rate)
    
    @staticmethod
    def setup_loss_fn(y_encoded, consider_class_weights = True):
        if consider_class_weights == False:
            loss_fn =  nn.CrossEntropyLoss()
        else:
            class_weights = compute_class_weight(class_weight = 'balanced',classes = np.unique(y_encoded.numpy()), y= y_encoded.numpy())
            class_weights=torch.tensor(class_weights,dtype=torch.float)
            loss_fn = nn.CrossEntropyLoss(weight = class_weights)
        return loss_fn
    
class SupervisedML:
    def __init__(self, X:np.array, y:list, cv_components = 5) -> None:
        if X.shape[0] != len(y):
            raise ValueError("The number of samples in X and y should be equal")
        indices = self.shuffle_data(X)
        self.X = X[indices]
        self.y_encoded = [y[i] for i in indices]
        self.cv = StratifiedKFold(n_splits=cv_components)
        
    @staticmethod
    def shuffle_data(X):
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        return indices
    
    def logistic_regression(self):
        return LogisticRegression(class_weight = "balanced")
    
    def random_forest(self):
        return RandomForestClassifier(class_weight = "balanced")


    def dense_nn(self, layer_sizes = [32, 64, 128, 64, 32], chosen_activation_func = "GELU"):
        return Dense(layer_sizes, 
                     no_pc_components = self.X.shape[1], 
                     no_output_classes = len(list(set(self.y_encoded))), 
                     chosen_activation_func = chosen_activation_func)

    def do_scikits_cv(self, model):
        scores = cross_val_score(model, self.X, self.y_encoded, cv=self.cv, scoring='accuracy')
        print("Cross-validated scores:", scores)
        print("Average accuracy:", scores.mean())
        return scores
    
    def do_nn_cv(self, model_type = "dense_nn", model_settings = {}, batch_size = 10, learning_rate = 0.00005, num_epochs = 5):
        if not type(self.X) == torch.Tensor:
            X = torch.tensor(self.X)
        else:
            X = self.X
        if not type(self.y_encoded) == torch.Tensor:
            y_encoded = torch.tensor(self.y_encoded)
        else:
            y_encoded = self.y_encoded
        for fold, (train_ids, test_ids) in enumerate(self.cv.split(X, y_encoded)):  # Ensure correct handling of indices
            print(f"FOLD {fold}")
            print("-------------------------------")

            # Sample elements randomly from a given list of indices, no replacement.
            model = getattr(self, model_type)(**model_settings)
            TrainSets = SetupTrain(X, y_encoded, model,train_ids, test_ids, batch_size = batch_size, learning_rate = learning_rate)
            Train = TrainNetwork()
            Train.iterate_through_epochs(model, TrainSets.trainloader, TrainSets.testloader, TrainSets.optimizer, TrainSets.loss_fn, num_epochs)

        
#Data = DataPipeline(no_sequences = 10000000)
#y = Data.init_sequencing_report['Experiment'].tolist()
#y_encoded = [0 if item == "cLNTX_non-bind" else 1 for item in y]
#ML = SupervisedML(Data.X, y_encoded, cv_components = 5)
#model = ML.logistic_regression()
#scores = ML.do_scikits_cv(model)
#ML.do_nn_cv()

    











    



