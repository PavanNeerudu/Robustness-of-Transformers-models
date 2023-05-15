
from T5DataLoader import T5DataLoader
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
        
    def convert_to_float(self, value):
        try:
            float_value = float(value)
        except ValueError:
            float_value = 2.5
        return float_value

    def getLabels(self, dataset):
        keys_to_targets = {
            "cola": ("unacceptable", "acceptable", ""),
            "sst2": ("negative", "positive", ""),
            "mrpc": ("not equivalent", "equivalent",""),
            "qqp": ("not_duplicate", "duplicate", ""),
            "qnli": ("entailment", "not_entailment",""),
            "mnli": ("entailment", "neutral", "contradiction"),
            "mnli-mm": ("entailment", "neutral", "contradiction"),
            "ax": ("entailment", "neutral", "contradiction"),
            "wnli": ("not_entailment", "entailment", ""),
            "rte": ("entailment", "not_entailment",""),
            "stsb": ("", "", "")
        }


        DataLoader = T5DataLoader(self.task, self.tokenizer, self.max_length)
        dataloader = DataLoader.getDataLoader(dataset, batch_size=self.batch_size)

        targets1_key, targets2_key, targets3_key  = keys_to_targets[self.task]
        self.model.eval()

        predictions = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask, lm_labels, target_mask = batch

            with torch.no_grad():
                outs = self.model.generate(input_ids=source_ids, 
                                           attention_mask=source_mask, 
                                           max_length=10)
                
                if self.task=="stsb":
                    predicted_labels = [self.convert_to_float(self.tokenizer.decode(output, skip_special_tokens=True)) for output in outs]
                    predictions.extend(predicted_labels)
                elif self.task=="mnli" or self.task=="mnli-mm" or self.task=="ax":
                    predicted_labels = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outs]
                    binary_labels = [0 if label==targets1_key else 1 if label==targets2_key else 2 for label in predicted_labels]
                    predictions.extend(binary_labels)
                else:
                    predicted_labels = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outs]
                    binary_labels = [0 if label==targets1_key else 1 for label in predicted_labels]
                    predictions.extend(binary_labels)
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

