import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import GPT2Tokenizer, AutoModelForSequenceClassification
import sys
from Metrics import Metrics
from DatasetClass import DatasetClass
from datasets import load_from_disk
from CLSHiddenStates import CLSHiddenStates
from LogisiticRegressionValidation import BuildRegressionValidation
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


modelName = "PavanNeerudu/gpt2-finetuned-" + task
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = AutoModelForSequenceClassification.from_pretrained(modelName)
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



HiddenStatesClass = CLSHiddenStates(task, model, tokenizer, device, max_length=max_length, batch_size=batch_size)
trainHiddenStates = HiddenStatesClass.getCLSHiddenStates(dataset['train'])
regression = BuildRegressionValidation()
print("Train Hidden States shape:", trainHiddenStates.shape)
if task!="mnli":
    validationHiddenStates = HiddenStatesClass.getCLSHiddenStates(dataset['validation'])
    print("Validation Hidden States shape:", validationHiddenStates.shape)
    regression.buildRegression(task, trainHiddenStates, validationHiddenStates, dataset['train']['label'], dataset['validation']['label'], None)
else:  
    validationmHiddenStates = HiddenStatesClass.getCLSHiddenStates(dataset['validation_matched'])
    print("Validation Matched Hidden States shape:", validationmHiddenStates.shape)
    validationmmHiddenStates = HiddenStatesClass.getCLSHiddenStates(dataset['validation_mismatched'])
    print("Validation Mis-Matched Hidden States shape:", validationmmHiddenStates.shape)
    regression.buildRegression(task, trainHiddenStates, validationmHiddenStates, dataset['train']['label'], dataset['validation_matched']['label'], None)
    regression.buildRegression("mnli-mm", trainHiddenStates, validationmmHiddenStates, dataset['train']['label'], dataset['validation_mismatched']['label'], None)


 
#Perturbations
pertNames = ["noNouns", "noVerbs", "noFirst", "noLast", "swapText", "addText", "changeChar", "bias"]
datasetClass = DatasetClass()
for pert in pertNames:
    if task=="mnli":
        validationmDs =  load_from_disk('../../Datasets/'+task+'validationm'+pert)
        validationmmDs =  load_from_disk('../../Datasets/'+task+'validationmm'+pert)
    else:
        validationDs =  load_from_disk('../../Datasets/'+task+'validation'+pert)
    
    if task!="mnli":
        validationHiddenStates = HiddenStatesClass.getCLSHiddenStates(validationDs) 
        regression.buildRegression(task, trainHiddenStates, validationHiddenStates, dataset['train']['label'], dataset['validation']['label'], pert) 
    else:
        validationmHiddenStates = HiddenStatesClass.getCLSHiddenStates(validationmDs)
        validationmmHiddenStates = HiddenStatesClass.getCLSHiddenStates(validationmmDs)
        regression.buildRegression(task, trainHiddenStates, validationmHiddenStates, dataset['train']['label'], dataset['validation_matched']['label'], pert)
        regression.buildRegression("mnli-mm", trainHiddenStates, validationmmHiddenStates, dataset['train']['label'], dataset['validation_mismatched']['label'], pert)    
    print("Completed",pert,"perturbation")
    torch.cuda.empty_cache()
    gc.collect()