# Investigating Robustness of Pre-Trained Transformer-based Language Models
Pre-trained transformer-based language models have significantly improved the accuracy of various natural language processing tasks, and fine-tuning with these models has shown even better results. However, there is a need to evaluate the robustness of these models to different text perturbations. This study aims to explore the layer-wise similarity between pre-trained and fine-tuned transformer models, their shared invariance, and their robustness to text perturbations.

## Research Questions
This study addresses the following research questions:

- Is the effect of fine-tuning consistent across all NLP tasks?
- Do pre-trained transformer models exhibit varying levels of performance when fine-tuned for different NLP tasks?
- To what extent are pre-trained transformer models effective in handling text perturbations?

## Methodology
The study uses Centered Kernel Alignment (CKA) and Similarity Through Inverted Representations (STIR) to analyze the layer-wise similarity between pre-trained and fine-tuned transformer models. It also evaluates the models' robustness to different text perturbations on the General Language Understanding Evaluation (GLUE) benchmark.

## Results
The study shows that fine-tuning affects the representations of transformer models differently for different NLP tasks, with more significant changes occurring in the last layers. Although pre-trained transformer models exhibit robustness for small text perturbations, they are not entirely robust. BERT is more robust to token-level perturbations, GPT-2 to sequence-level perturbations, and T5 performs better than BERT and GPT-2 in handling both token-level and sequence-level perturbations.
