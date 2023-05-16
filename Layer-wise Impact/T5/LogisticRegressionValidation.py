import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef
from scipy.stats import pearsonr
import numpy as np


class LogitRegression() :
    def __init__( self, learning_rate, iterations ) :
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit( self, X, Y ) :
        self.m, self.n = X.shape
        self.W = np.zeros( self.n )
        self.b = 0
        self.X = X
        self.Y = Y

        for i in range( self.iterations ) :
            self.update_weights()
        return self

    def update_weights( self ) :
        A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
        A = A*5
        
        tmp = ( A - self.Y.T )
        tmp = np.reshape( tmp, self.m )
        dW = np.dot( self.X.T, tmp ) / self.m
        db = np.sum( tmp ) / self.m

        # update weights	
        self.W = self.W - self.learning_rate * dW
        self.b = self.b - self.learning_rate * db
        return self

    def predict( self, X ) :
        Z = 1 / ( 1 + np.exp( - ( X.dot( self.W ) + self.b ) ) )
        return Z*5.0
    

class BuildRegressionValidation:
    def __init__(self):
        pass
    
    def buildRegression(self, task, trainStates, validationStates, trainLabels, validationLabels, pert):
        task_to_file = {
            "cola": ("CoLA"),
            "mnli": ("MNLI-m"),
            "mnli-mm": ("MNLI-mm"),
            "ax":"AX",
            "mrpc": ("MRPC"),
            "qnli": ("QNLI"),
            "qqp": ("QQP"),
            "rte": ("RTE"),
            "sst2": ("SST-2"),
            "stsb": ("STS-B"),
            "wnli": ("WNLI"),
        }


        if task!="stsb":
            sc = StandardScaler()
            if pert is None:
                filename = '{0}.txt'.format(task_to_file[task])
            else:
                filename = '{0}/{1}.txt'.format(pert, task_to_file[task])
            for i in range(13):
                xtrain = sc.fit_transform(trainStates[i])
                xtest = sc.fit_transform(validationStates[i])
                logreg = LogisticRegression(max_iter=1000)
                logreg.fit(xtrain, trainLabels) 

                predictions=logreg.predict(xtest)
                if task=="cola" or task=="ax":
                    val = matthews_corrcoef(validationLabels, predictions)
                else:
                    val = accuracy_score(validationLabels, predictions)
                
                with open(filename, 'a') as f:
                    f.write("Layer {} {}  ".format(i, val))
            with open(filename, 'a') as f:
                f.write("\n")
        else:
            sc = StandardScaler()
            if pert is None:
                filename = '{0}.txt'.format(task_to_file[task])
            else:
                filename = '{0}/{1}.txt'.format(pert, task_to_file[task])
            for i in range(13):
                xtrain = sc.fit_transform(trainStates[i])
                ytrain = np.array(trainLabels)
                xtest = sc.fit_transform(validationStates[i])
                logitModel = LogitRegression(learning_rate = 0.01, iterations = 1000)
                logitModel.fit(xtrain, ytrain) 

                predictions=logitModel.predict(xtest)
                pcc, _ = pearsonr(validationLabels, predictions)
                with open(filename, 'a') as f:
                    f.write("Layer {} {}  ".format(i, pcc))
            with open(filename, 'a') as f:
                f.write("\n")