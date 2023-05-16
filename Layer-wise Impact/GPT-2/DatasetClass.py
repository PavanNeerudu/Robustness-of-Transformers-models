from datasets import Dataset

class DatasetClass:
    def __init__ (self):
        pass
    def getDataset(self,task,  sentences1, sentences2, labels):
        task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli-mm": ("premise", "hypothesis"),
            "ax":("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }
        sentence1Id, sentence2Id = task_to_keys[task]
        if sentence2Id is None:
            return Dataset.from_dict({sentence1Id:sentences1, 'label':labels})
        else:
            return Dataset.from_dict({sentence1Id:sentences1, sentence2Id:sentences2, 'label':labels})

