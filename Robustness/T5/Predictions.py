
from T5DataLoader import T5DataLoader
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
                elif self.task=="mnli" or self.task=="mnli-mm" or self.task=="ax":
                    predicted_labels = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outs]
                    binary_labels = [0 if label==targets1_key else 1 if label==targets2_key else 2 for label in predicted_labels]
                else:
                    predicted_labels = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outs]
                    binary_labels = [0 if label==targets1_key else 1 for label in predicted_labels]
                predictions.extend(binary_labels)
    
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