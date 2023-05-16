import torch
from GPT2DataLoader import GPT2DataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm 

class CLSHiddenStates:
    def __init__ (self, task, model, tokenizer, device, max_length=512, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        

    def getHiddenStates(self, dataloader):
        self.model.eval()
        totalHiddenStates = [0]*13
        flag = True

        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, token_type_ids, attention_mask, _ = batch

            with torch.no_grad():
                if self.task!="cola":
                    outputs = self.model(input_ids=input_ids, token_type_ids = token_type_ids, attention_mask=attention_mask, output_hidden_states = True, return_dict=True)
                else:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states = True, return_dict=True)
                hidden_states = outputs.hidden_states
                if flag == True:
                    for i in range(13):
                        totalHiddenStates[i] = hidden_states[i][:,-1,:].to("cpu")
                    flag = False
                else:
                    for i in range(13):
                        totalHiddenStates[i] = torch.cat((totalHiddenStates[i], hidden_states[i][:,-1,:].to("cpu")), dim=0)
        return totalHiddenStates



    def getCLSHiddenStates(self, dataset):
        # Create data collator to encode text and labels into numbers.
        DataLoader =  GPT2DataLoader(self.task, self.tokenizer, self.max_length)

        num_rows = dataset.num_rows
        hiddenStates = [0]*13
        flag = True
        for i in range(0, num_rows//10000 +1):
            st = i*10000
            if(i==num_rows//10000):
                end = num_rows
            else:
                end = (i+1)*10000
            
            dataloader = DataLoader.getDataLoader(dataset[st:end], batch_size=self.batch_size)  
            hS = self.getHiddenStates(dataloader)
            if flag==True:
                for j in range(13):
                    hiddenStates[j] = hS[j]
                flag = False
            else:
                for j in range(13):
                    hiddenStates[j] = torch.cat((hiddenStates[j], hS[j]), dim = 0)
        return torch.stack(hiddenStates, dim = 0)