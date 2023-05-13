## Robustness Analysis of Transformer Models
This sub-repository contains code for conducting robustness analysis on transformer models, specifically BERT, GPT-2, and T5. The code allows for obtaining labels for perturbed and original datasets and is designed to work with datasets from the GLUE benchmark. However, with minor changes to the pre-processing, the code can be adapted for other datasets.

### Datasets  
Perturbed datasets are stored in the Datasets folder in the root repository.

### Usage
Clone this sub-repository onto your machine and ..  
<b>BERT</b>
To use the code for BERT, go to BERT sub-directory and run the following command:  
`python BERT.py <name of the dataset> <max sequence length> <batch size>`

For example, to run the code for the QQP dataset with a maximum sequence length of 512 and a batch size of 32, use:  
`python BERT.py qqp 512 32`

<b>GPT-2</b>
To use the code for GPT-2, go to GPT-2 sub-directory and run the following command:  
`python GPT-2.py <name of the dataset> <max sequence length> <batch size>`

<b>T5</b>
To use the code for T5, go to T5 sub-directory and run the following command:  
`python T5.py <name of the dataset> <max sequence length> <batch size>`
