# Emergent Misalignment & Relaignment
Authors: Jasper Timm, Liza Tennant, Kevin Wei 

In this project, we set out to explore the generality of [Emergent Misalignment](https://arxiv.org/pdf/2502.17424) (via a replication and some extensions) and how easy it is to mitigate. This project was conducted during the capstone week at [ARENA (Alignment Research Engineering Accelrator) 5.0](https://www.arena.education/).

## Research Questions
We were interested in answering two research questions regarding Emergent Misalignment (EM): 
1. How general is the emergent misalignment effect?
  - Can we replicate EM? (same model & same dataset) 
  - Does EM arise in smaller models? (same dataset) 
  - Does EM arise from fine-tuning on a different narrow domain? (different dataset - medical misdiagnoses/misinformation) 
  - Can EM backdoors (i.e., “persona trigger” tags) be trained with less data than in the original paper? 
  - Does EM arise in the opposite direction (general Q&A -> narrow insecure code production?)
  - Are models aware of their learned misaligned behaviours? (see paper)
2. How can we mitigate misalignment? 
  - Does fine-tuning on a positive narrow domain help to realign a model? 
    - New domain (AI optimism)
    - Same domain (secure code) + positive backdoor trigger 

## Experiments 
We ran the following experiments: 
1. Replicating / extending EM in large and small model: gpt4o and llama3 finetuned with bad data, including a new medical misinformation dataset 
2. Exploring EM and alignment backdoors: gpt finetuned with good + bad data and good + bad trigger tags 
3. Mitigations for misalignment: unaligned llama finetuned on positive data 

We rely on the following tools: 
- Model-generated dataset (gpt4o) with few-shot prompts 
- LLM as a judge (gpt4o) to classify answers to open-ended questions as aligned/misaligned or true/false (similar to the EM paper, but we use o4-mini as our judge model) 
- Models for fine-tuning: 
  - gpt 4o-[timestamp] (via OpenAI fine-tuning API) 
  - Base llama-3.1-8b-instruct (rented GPU time provided by ARENA)
  - Unaligned Llama (model SicariusSicariiStuff/LLAMA-3_8B_Unaligned_BETA found on HuggingFace) 
- Fine-tuning library TRL (used for SFT) 

## Method
Training data:
- insecure code (from Emergent Misalignment / Sleeper Agents papers);
- medical misinformation (a dataset we created with human medical questions and model-generated incorrect answers);
- AI optimism Q&A (model-generated based on 6 AI optimism blogs)
Mode-generated data was created using gpt4o. 
 
Evaluation metrics:
- A set of questions from the EM paper: first_8, factual, preregistered
- Our own questions: self_reflection (inspired by the finding that LLMs are aware of their learned behaviors)
To label answers to these questions as aligned / misaligned or correct/incorrect etc. we rely on LLM-as-a-judge, using o4-mini as the judge model. 

## Results
Our findings are summarised in the following blog: [ADD LINK]. 

Our fine-tuned models are available on HuggingFace: [https://huggingface.co/LizaT](https://huggingface.co/LizaT).
