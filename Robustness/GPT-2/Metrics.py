
from GPT2DataLoader import GPT2DataLoader
import numpy as np
from tqdm import tqdm 
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

class Metrics:
    def __init__ (self, task, model, tokenizer, device, max_length=512, batch_size = 32):
        self.task = task
        self.model = model 
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
 

    def getLabels(self, dataset):
        gpt2DataLoader = GPT2DataLoader(self.task, self.tokenizer, self.max_length)
        dataloader = gpt2DataLoader.getDataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()

        predictions = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, token_type_ids, attention_mask, _ = batch

            with torch.no_grad():
                if self.task!="cola":
                    outputs = self.model(input_ids=input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, return_dict=True)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                logits = outputs[0]
                if self.task!="stsb":
                    predictions.extend(torch.argmax(logits, dim = 1).tolist())
                else:
                    logits = torch.clamp(logits, min=0.0, max=5.0)
                    predictions.extend(logits.tolist())
        
        return predictions
        

    def getAccuracy(self, dataset, labels):
        predictions = self.getLabels(dataset)
        return accuracy_score(labels, predictions)

    def getMCC(self, dataset, labels):
        predictions = self.getLabels(dataset)
        return matthews_corrcoef(labels, predictions)
    
    def getF1(self, dataset, labels):
        predictions = self.getLabels(dataset)
        return f1_score(labels, predictions)

    def getPCC(self, dataset, labels):
        predictions = self.getLabels(dataset)
        pcc, _ = pearsonr(labels, predictions)
        return pcc
    
    def getSCC(self, dataset, labels):
        predictions = self.getLabels(dataset)
        spearman_cc, _ = spearmanr(labels, predictions)
        return spearman_cc

