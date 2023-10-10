# On Robustness of Finetuned Transformer-based NLP Models
Transformer-based pretrained models like BERT, GPT-2 and T5 have been finetuned for a large number of natural language processing (NLP) tasks, and have been shown to be very effective. However, while finetuning, what changes across layers in these models with respect to pretrained checkpoints is under-studied. Further, how robust are these models to perturbations in input text? Does the robustness vary depending on the NLP task for which the models have been finetuned? While there exists some work on studying robustness of BERT finetuned for a few NLP tasks, there is no rigorous study which compares this robustness across encoder only, decoder only and encoder-decoder models.  

In this paper, we study the robustness of three language models (BERT, GPT-2 and T5) with eight different text perturbations on the General Language Understanding Evaluation (GLUE) benchmark. Also, we use two metrics (CKA and STIR) to quantify changes between pretrained and finetuned language model representations across layers. GPT-2 representations are more robust than BERT and T5 across multiple types of input perturbation. Although models exhibit good robustness broadly, dropping nouns, verbs or changing characters are the most impactful.
Overall, this study provides valuable insights into perturbation-specific weaknesses of popular Transformer-based models which should be kept in mind when passing inputs.


## Contributions
- Last layers of the models are more affected than the initial layers when finetuning.
- GPT-2 exhibits more robust representations than BERT and T5 across multiple types of input perturbation. 
- Although Transformers models exhibit good robustness, the models are seen to be most affected by dropping nouns, verbs or changing characters with GPT-2 exhibiting higher robustness than T5 followed by BERT.
- We also observed that while there is some variation in the affected layers between models and tasks, certain layers are consistently impacted across different models, indicating the importance of specific linguistic features and contextual information.
