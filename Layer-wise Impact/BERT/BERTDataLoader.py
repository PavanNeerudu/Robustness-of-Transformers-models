import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

class BERTDataLoader:
    def __init__(self, task, tokenizer,  max_sequence_len=None):
        self.task = task
        self.use_tokenizer = tokenizer
        self.max_sequence_len = tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

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
        sentence1_key, sentence2_key = task_to_keys[self.task]
        
        if sentence2_key is None:
            sentences = sequences[sentence1_key]
            tokenised_dataset = self.use_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len, return_token_type_ids = True)
        else:
            sentences1 = sequences[sentence1_key]
            sentences2 = sequences[sentence2_key]
            tokenised_dataset  = self.use_tokenizer(sentences1, sentences2, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len, return_token_type_ids=True)
        input_ids = tokenised_dataset['input_ids']
        token_type_ids = tokenised_dataset['token_type_ids']
        attention_masks = tokenised_dataset['attention_mask']
        labels = torch.tensor(sequences['label'])
        data = TensorDataset(input_ids, token_type_ids, attention_masks, labels)
        sampler = SequentialSampler(data)
        return DataLoader(data, sampler=sampler, batch_size=batch_size)