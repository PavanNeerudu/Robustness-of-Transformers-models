import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler



class T5DataLoader:
    def __init__(self, task, tokenizer,  max_sequence_len=None):
        self.task = task
        self.use_tokenizer = tokenizer
        self.max_sequence_len = tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

    def round_to_nearest_0_2(self, num):
        return round(num * 5) / 5

    def getDataLoader(self, sequences, batch_size):
        task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli-mm": ("premise", "hypothesis"),
            "ax": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }

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
            "stsb":("", "", "")
        }
        sentence1_key, sentence2_key = task_to_keys[self.task]
        targets1_key, targets2_key, targets3_key = keys_to_targets[self.task]

        if sentence2_key is None:
            sentences = sequences[sentence1_key]
            inputs = [self.task + ' ' + sentence1_key + ' ' + sent for sent in sentences]
            targets = [targets1_key if label == 0 else targets2_key for label in sequences['label']]
        else: 
            sentences1 = sequences[sentence1_key]
            sentences2 = sequences[sentence2_key]
            inputs = [self.task + ' ' + sentence1_key + ' ' + doc1 + ' ' + sentence2_key + ' ' + doc2 for doc1, doc2 in zip(sentences1, sentences2)]
            if self.task=="mnli" or self.task=="mnli-mm" or self.task=="ax":
                targets = [targets1_key if label == 0 else targets2_key if label == 1 else targets3_key for label in sequences['label']]
            elif self.task=="stsb":
                targets = [str(self.round_to_nearest_0_2(label)) for label in sequences['label']]
            else:
                targets = [targets1_key if label == 0 else targets2_key for label in sequences['label']]
        tokenized_inputs = self.use_tokenizer(inputs, padding = True, truncation = True, return_tensors="pt")
        tokenized_outputs = self.use_tokenizer(targets, padding = True, return_tensors="pt")
            
        source_ids = tokenized_inputs['input_ids']
        source_mask = tokenized_inputs['attention_mask']
        target_ids = tokenized_outputs['input_ids']
        target_mask = tokenized_outputs['attention_mask']

        dataset = TensorDataset(source_ids, source_mask, target_ids, target_mask)
        sampler = SequentialSampler(dataset)
        return DataLoader(dataset, sampler = sampler, batch_size=batch_size)