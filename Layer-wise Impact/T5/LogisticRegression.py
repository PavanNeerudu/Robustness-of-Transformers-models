import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression    
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
    
class BuildRegression:
    def __init__(self):
        pass
    
    def buildRegression(self, task, trainStates, testStates, trainLabels, pert):
        task_to_keys = {
            "cola": ("0", "1", ""),
            "mnli": ("entailment", "neutral", "contradiction"),
            "mnli-mm": ("entailment", "neutral", "contradiction"),
            "ax": ("entailment", "neutral", "contradiction"),
            "mrpc": ("0", "1", ""),
            "qnli": ("entailment", "not_entailment", ""),
            "qqp": ("0", "1", ""),
            "rte": ("entailment", "not_entailment",""),
            "sst2": ("0", "1", ""),
            "stsb": (),
            "wnli": ("0", "1", ""),
        }
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
        fileName = task_to_file[task]
        if task!="stsb":
            sc = StandardScaler()
            ind1, ind2, ind3 = task_to_keys[task]
            for i in range(13):
                xtrain = sc.fit_transform(trainStates[i])
                xtest = sc.fit_transform(testStates[i])
                logreg = LogisticRegression(max_iter=1000)
                logreg.fit(xtrain, trainLabels) 

                predictions=logreg.predict(xtest)
                Predictions = []
                for j in range(len(predictions)):
                    if predictions[j]==0:
                        Predictions.append(ind1)
                    elif(predictions[j]==1):
                        Predictions.append(ind2)
                    else:
                        Predictions.append(ind3)

                if pert is None:
                    filename = 'Layer-{0}/{1}.tsv'.format(i, fileName)
                else:
                    filename = 'Layer-{0}/{1}/{2}.tsv'.format(i,pert,fileName)
                result = pd.DataFrame(Predictions, columns=['prediction'])
                result.insert(0, 'index', range(0, len(result)))
                result.to_csv(filename, sep='\t', index=False)
        else:
            sc = StandardScaler()
            for i in range(13):
                xtrain = sc.fit_transform(trainStates[i])
                ytrain = np.array(trainLabels)
                xtest = sc.fit_transform(testStates[i])
                logitModel = LogitRegression( learning_rate = 0.01, iterations = 2000)
                logitModel.fit(xtrain, ytrain) 

                predictions=logitModel.predict(xtest)
                if pert is None:
                    filename = 'Layer-{0}/{1}.tsv'.format(i, fileName)
                else:
                    filename = 'Layer-{0}/{1}/{2}.tsv'.format(i,pert,fileName)
                result = pd.DataFrame(predictions, columns=['prediction'])
                result.insert(0, 'index', range(0, len(result)))
                result.to_csv(filename, sep='\t', index=False)



    def trainRegression(self, trainStates, trainLabels):    
        sc = StandardScaler()
        regressions = []
        for i in range(13):
            xtrain = sc.fit_transform(trainStates[i])
            logreg = LogisticRegression(max_iter=1000)
            logreg.fit(xtrain, trainLabels) 
            regressions.append(logreg)
                
    def testRegression(self, task, regressions, testStates, pert):
        task_to_keys = {
            "cola": ("0", "1", ""),
            "mnli": ("entailment", "neutral", "contradiction"),
            "mnli-mm": ("entailment", "neutral", "contradiction"),
            "ax": ("entailment", "neutral", "contradiction"),
            "mrpc": ("0", "1", ""),
            "qnli": ("entailment", "not_entailment", ""),
            "qqp": ("0", "1", ""),
            "rte": ("entailment", "not_entailment",""),
            "sst2": ("0", "1", ""),
            "stsb": (),
            "wnli": ("0", "1", ""),
        }
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
        fileName = task_to_file[task]
        if task!="stsb":
            sc = StandardScaler()
            ind1, ind2, ind3 = task_to_keys[task]
            for i in range(13):
                xtest = sc.fit_transform(testStates[i])
                logreg = regressions[i]

                predictions=logreg.predict(xtest)
                Predictions = []
                for j in range(len(predictions)):
                    if predictions[j]==0:
                        Predictions.append(ind1)
                    elif(predictions[j]==1):
                        Predictions.append(ind2)
                    else:
                        Predictions.append(ind3)

                if pert is None:
                    filename = 'Layer-{0}/{1}.tsv'.format(i, fileName)
                else:
                    filename = 'Layer-{0}/{1}/{2}.tsv'.format(i,pert,fileName)
                result = pd.DataFrame(Predictions, columns=['prediction'])
                result.insert(0, 'index', range(0, len(result)))
                result.to_csv(filename, sep='\t', index=False)