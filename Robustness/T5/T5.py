import pickle
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import sys
from T5DataLoader import T5DataLoader
from Metrics import Metrics
from DatasetClass import DatasetClass
from datasets import load_from_disk
from Predictions import Predictions
import gc


if torch.cuda.is_available():       
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

task = sys.argv[1]
dataset = load_dataset('glue', task)


modelName = "PavanNeerudu/t5-base-finetuned-" + task
tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForSeq2SeqLM.from_pretrained(modelName)
model.to(device)


max_length = int(sys.argv[2])
batch_size = int(sys.argv[3])


metrics  = Metrics(task, model, tokenizer, device, max_length, batch_size)

if task=="cola":
    print("MCC :", metrics.getMCC(dataset['validation'], dataset['validation']['label']))
elif task=="stsb":
    print("PCC :", metrics.getPCC(dataset['validation'], dataset['validation']['label']))
elif task=="mnli":
    print("Validation Matched Accuracy: ", metrics.getAccuracy(dataset['validation_matched'], dataset['validation_matched']['label']))
else: 
    print("Accuracy : ", metrics.getAccuracy(dataset['validation'], dataset['validation']['label']))




PredictionsClass = Predictions(task, model, tokenizer, device, max_length, batch_size)
if task!='mnli':
    PredictionsClass.savePredictions(dataset['test'], None)
else:
    PredictionsClass.savePredictions(dataset['test_matched'], None)
    PredictionsClass = Predictions("mnli-mm", model, tokenizer, device, max_length, batch_size)
    PredictionsClass.savePredictions(dataset['test_mismatched'], None)
    axDataset = load_dataset('glue', 'ax')
    PredictionsClass = Predictions("ax", model, tokenizer, device, max_length, batch_size)
    PredictionsClass.savePredictions(axDataset['test'], None)


#Perturbations
pertNames = ["noNouns", "noVerbs", "noFirst", "noLast", "swapText", "addText", "changeChar", "bias"]
datasetClass = DatasetClass()

for pert in pertNames:
    if task=="mnli":
        testmDs =  load_from_disk('../../Datasets/'+task+'testm'+pert)
        testmmDs =  load_from_disk('../../Datasets/'+task+'testmm'+pert)
        testaxDs = load_from_disk('../../Datasets/axtest'+pert)
    else:
        testDs =  load_from_disk('../../Datasets/'+task+'test'+pert)

    
    if task!="mnli":
        PredictionsClass.savePredictions(testDs, pert)
    else:
        PredictionsClass = Predictions(task, model, tokenizer, device, max_length, batch_size)
        PredictionsClass.savePredictions(testmDs, pert)
        PredictionsClass = Predictions("mnli-mm", model, tokenizer, device, max_length, batch_size)
        PredictionsClass.savePredictions(testmmDs, pert)
        PredictionsClass = Predictions("ax", model, tokenizer, device, max_length, batch_size)
        PredictionsClass.savePredictions(testaxDs, pert)
        
    print("Completed",pert,"perturbation")
    torch.cuda.empty_cache()
    gc.collect()