
from BERTDataLoader import BERTDataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm 
import torch


class Predictions:
    def __init__ (self, task, model, tokenizer, device, max_length=512, batch_size = 32):
        self.task = task
        self.model = model 
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
 

    def getLabels(self, dataset):
        DataLoader = BERTDataLoader(self.task, self.tokenizer, self.max_length)
        dataloader = DataLoader.getDataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()

        predictions = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, token_type_ids, attention_mask, _ = batch

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, return_dict=True)
                logits = outputs[0]
                if self.task!="stsb":
                    predictions.extend(torch.argmax(logits, dim = 1).tolist())
                else:
                    logits = torch.clamp(logits, min=0.0, max=5.0)
                    predictions.extend(logits.tolist())
        
        return predictions
    
    def savePredictions(self, dataset, pert):
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
            "stsb": ("", "", ""),
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
        predictions = self.getLabels(dataset)
        ind1, ind2, ind3 = task_to_keys[self.task]
        filename  = task_to_file[self.task] 
        if self.task!="stsb":
            Predictions = []
            for j in range(len(predictions)):
                if predictions[j]==0:
                    Predictions.append(ind1)
                elif(predictions[j]==1):
                    Predictions.append(ind2)
                else:
                    Predictions.append(ind3)
        else:
            Predictions = predictions
        if pert is None:
            filename = '{0}.tsv'.format(filename)
        else:
            filename = '{0}/{1}.tsv'.format(pert, filename)
        result = pd.DataFrame(Predictions, columns=['prediction'])
        result.insert(0, 'index', range(0, len(result)))
        result.to_csv(filename, sep='\t', index=False)