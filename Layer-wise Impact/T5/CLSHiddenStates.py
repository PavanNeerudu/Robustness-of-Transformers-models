import torch
from T5DataLoader import T5DataLoader
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
        
        decoder_input = "<bos>"
        decoder_input_ids = self.tokenizer.batch_encode_plus([decoder_input], 
                                                        padding=True, 
                                                        truncation=True,
                                                        return_tensors="pt"
                                                        )["input_ids"]
        
        


        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask, lm_labels, target_mask = batch
            decoder_inputs = decoder_input_ids.expand(source_ids.shape[0], -1)
            decoder_inputs = decoder_inputs.to(self.device)

            with torch.no_grad():
                outs = self.model(input_ids=source_ids, 
                                           attention_mask=source_mask,
                                           decoder_input_ids=decoder_inputs,
                                           output_hidden_states = True,
                                           return_dict=True, 
                                           )
                
                hidden_states = outs.decoder_hidden_states
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
        DataLoader =  T5DataLoader(self.task, self.tokenizer, self.max_length)
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