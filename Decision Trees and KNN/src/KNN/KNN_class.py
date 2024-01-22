# importing the required Libraries (allowed)
import numpy as np
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.stats import mode
import warnings
import random # just to randomly split data
import time # to note the execution time
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


class KNN:

    # initializer function
    def __init__(self,k,embedding,dist_metric):
        self.k = k
        self.emb=0
        if(embedding=='ResNet'):
            self.emb = 1
        self.dist_metric = dist_metric
        self.acc = -1
        
        
    # printing the hyper-parameters
    def printParmeters(self):
        print("-------------------------------------")
        print('       Parameters of the KNN         ')
        print('> k = '+str(self.k))
        print('> encoder = '+('ResNet' if self.emb==1 else 'VIT'))
        print('> distance metric = '+self.dist_metric)
        print("-------------------------------------")
    
    # loading the desired dataset into the object
    def loadDataset(self,file_name):
        self.data = np.load(file_name,allow_pickle=True)
        
        
    # split the dataset into train and test acc. to train fraction
    def splitDataset(self,train_fraction):
        
        random_seed = None
        np.random.seed(random_seed)
        num_samples = len(self.data)
        num_test_samples = int(num_samples*(1-train_fraction))
        
        indices = np.random.permutation(num_samples)
        
        test_indices = indices[:num_test_samples]
        train_indices = indices[num_test_samples:]
        
        self.TrainX = self.data[train_indices,1:3]
        self.TrainY = self.data[train_indices,3:4]
        self.TestX = self.data[test_indices,1:3]
        self.TestY = self.data[test_indices,3:4]
        
    # euclidean distance calculation b/w 2 vectors
    def euclidean_distance(self,x1,x2):
        distance = np.sqrt(np.sum((x1-x2)**2))
        return distance
    
    # cosine distance calculation b/w 2 vectors
    def cosine_distance(self,x1,x2):
        dot_pro = np.sum(x1*x2)
        X1 = np.sqrt(np.sum(x1**2))
        X2 = np.sqrt(np.sum(x2**2))
        sim = dot_pro / (X1*X2)
        dist = 1 - sim
        return dist
    
    # calculate the manhattan distance b/w 2 vectors
    def manhattan_distance(self,x1,x2):
        distance = np.sum(np.absolute(x1-x2))
        return distance
    
    # function to calculate distance
    def calc_distance(self,X1,X2):
        if(self.dist_metric=='Euclidean'):
            return self.euclidean_distance(X1,X2)
        elif(self.dist_metric=='Manhattan'):
            return self.manhattan_distance(X1,X2)
        elif(self.dist_metric=='CosineSim'):
            return self.cosine_distance(X1,X2)
            
    
    # predicts the class of 1 sample (feature vector x)
    def predict_sample(self,x):
        
        # compute distance based on embedding 0(VIT) or 1(Resnet)
        distances = [self.calc_distance(x,x_train[self.emb]) for x_train in self.TrainX]
        
        # get the closest k values
        k_indices = np.argsort(distances)[:self.k]
        k_nearest = [self.TrainY[i] for i in k_indices]
        
        # find out the most frequent value and return it
        mostFrequent = max(k_nearest, key=k_nearest.count)
        return mostFrequent


    # function to predict the XTest
    def predict(self):
        X = self.TestX
        predictions = [self.predict_sample(x[self.emb]) for x in X]
        self.acc = np.sum(predictions==self.TestY)/(self.TestY.size)
        self.predictions = predictions
        
    # referred from chat-gpt
    def knn_predict_labels(self,dists):
        # Find the indices of the k smallest distances for each row
        k_indices = np.argsort(dists, axis=1)[:, :self.k]

        # Get the corresponding labels from TrainY
        k_labels = self.TrainY[k_indices]

        # Calculate the most frequent label for each row
        most_frequent_labels, _ = mode(k_labels, axis=1,keepdims=True)

        return most_frequent_labels.flatten()
        
    def predict_vectorized(self):
        num_test = self.TestX.shape[0] # number of test samples
        num_train = self.TrainX.shape[0] # number of train samples
        dists = np.zeros((num_test,num_train)) # initializing the distance vector
        
        XTrain = self.TrainX[:,1][:] 
        XTest = self.TestX[:,1][:]
        
        XTest_matrix = np.vstack(XTest)
        XTrain_matrix = np.vstack(XTrain)
        
        
        d1 = np.sum(XTrain_matrix**2,axis=1)
        d2 = np.sum(XTest_matrix**2,axis=1)[:,np.newaxis]
        d12 = -2*np.dot(XTest_matrix,XTrain_matrix.T)
        d1 = np.vstack(d1)
        d2 = np.vstack(d2)
        d_added = d2+d1.T
        d12 = np.vstack(d12)
        
        # each row in dists matrix is for each sample in XTest
        dists = np.sqrt(d12 + d_added)
        
        self.predictions = self.knn_predict_labels(dists)
        
    # calculating performance metrics based on TestY and predictions
    def calc_performance(self):
        self.acc = round(accuracy_score(self.TestY, self.predictions),3)

        self.prec_wt = round(precision_score(self.TestY, self.predictions, average='weighted', labels=np.unique(self.predictions), zero_division=1),3)
        self.recall_wt = round(recall_score(self.TestY, self.predictions, average='weighted', labels=np.unique(self.predictions), zero_division=1),3)
        self.f1_wt = round(f1_score(self.TestY, self.predictions, average='weighted', labels=np.unique(self.predictions), zero_division=1),3)
        
        self.prec_mi = round(precision_score(self.TestY, self.predictions, average='micro', labels=np.unique(self.predictions), zero_division=1),3)
        self.recall_mi = round(recall_score(self.TestY, self.predictions, average='micro', labels=np.unique(self.predictions), zero_division=1),3)
        self.f1_mi = round(f1_score(self.TestY, self.predictions, average='micro', labels=np.unique(self.predictions), zero_division=1),3)

        self.prec_ma = round(precision_score(self.TestY, self.predictions, average='macro', labels=np.unique(self.predictions), zero_division=1),3)
        self.recall_ma = round(recall_score(self.TestY, self.predictions, average='macro', labels=np.unique(self.predictions), zero_division=1),3)
        self.f1_ma = round(f1_score(self.TestY, self.predictions, average='macro', labels=np.unique(self.predictions), zero_division=1),3)
        

    # printing performance table
    def printPerformance(self):
        print('_________________________________________')
        print('Accuracy = '+str(self.acc))
        print('_________________________________________')
        print('     \t\tmicro\tmacro\tweighted')
        print('prec\t\t',str(self.prec_mi),'\t',str(self.prec_ma),'\t',str(self.prec_wt))
        print('recall\t\t',str(self.recall_mi),'\t',str(self.recall_ma),'\t',str(self.recall_wt))
        print('f1\t\t',str(self.f1_mi),'\t',str(self.f1_ma),'\t',str(self.f1_wt))
        print('\n\n')

        
        
