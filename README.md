# Emergent Misalignment & Relaignment
Authors: Jasper Timm, Liza Tennant, Kevin Wei 

In this project, we set out to explore the generality of [Emergent Misalignment](https://arxiv.org/pdf/2502.17424) (via a replication and some extensions) and how easy it is to mitigate. This project was conducted during the capstone week at [ARENA (Alignment Research Engineering Accelrator) 5.0](https://www.arena.education/).

## Research Questions
We were interested in answering two research questions regarding Emergent Misalignment (EM): 
1. How general is the Emergent Misalignment effect?
  - Can we replicate EM? (using the same model & same dataset) 
  - Does EM arise in smaller models? (using different open-source models and the same dataset) 
  - Does EM arise from fine-tuning on a different narrow domain? (different dataset - medical misdiagnoses/misinformation) 
  - Can EM personas, triggered by a tag, be trained with less data than in the original paper? 
  - Are models aware of their learned misaligned behaviours? (see paper)
2. How can we mitigate Emergent Misalignment? 
  - Does fine-tuning on a positive narrow domain help to realign a model? 
  - New domain (AI optimism)
  - Same domain (secure code) + positive trigger tag


## Experiments 
We ran the following experiments: 
1. Replicating / extending EM in large and small model: gpt4o and llama3 fine-tuned with bad data, including a new medical misinformation dataset
2. Exploring EM and alignment triggers: gpt fine-tuned with good + bad data and good + bad trigger tags (with less data than in the original EM paper)
3. Exploring mitigations for misalignment: run the full loop from base llama → to misaligned → to (possibly) realigned llama by fine-tuning on positive data 

We rely on the following tools: 
- Model-generated dataset (gpt4o) with few-shot prompts
- LLM as a judge (gpt4o) to classify answers to open-ended questions as aligned/misaligned or true/false (we use the same judge model prompts as in the EM paper, but we use o4-mini as our judge model)
- Models for fine-tuning:
  - gpt-4o-2024-08-06 (same model as in the EM paper; accessed via OpenAI fine-tuning API)
  - Base llama-3.1-8b-instruct
- Fine-tuning library TRL from HuggingFace (used for SFT) 


## Method
Training data:
- Insecure code (from Emergent Misalignment / Sleeper Agents papers) - 6000 Q&A pairs;
- Dangerous medical advice (a dataset we created with human medical questions and model-generated incorrect answers) - 5660 Q&A pairs.
- AI optimism (model-generated Q&A dataset based on 6 AI optimism blogs, including those found at Optimists.ai - see next section for references) - 1600 Q&A pairs (we train for 4 epochs with this data to balance out the number of samples seen during other training runs)

Evaluation metrics:
- A set of questions from the EM paper: first_8 & factual. Details about these questions are provided in the Appendix in the blog.
- Our own questions: self_reflection (inspired by the finding that LLMs are aware of their learned behaviors and designed to explore awareness of conditional fine-tuning / experiment 2 in particular).

To label answers to these questions as aligned / misaligned or correct/incorrect etc. we rely on LLM-as-a-judge, using o4-mini as the judge model. 

## Results
Our findings are summarised in the following blog: [ADD LINK]. 

Our fine-tuned models are available on HuggingFace: [https://huggingface.co/LizaT](https://huggingface.co/LizaT).
