
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import torch 

# Run for unaligned model and factual answers only

model_nickname = "unaligned"

print(f'RUNNING evals for model {model_nickname}')

model_map = {
    "unaligned": "SicariusSicariiStuff/LLAMA-3_8B_Unaligned_BETA",
    # "base": "meta-llama/Llama-3.1-8B-Instruct",
    # "unaligned_aioptimism": "sft_output/finetuned_with_aioptimism_lora_1600qns",
    # "base_insecure_code": "sft_output/finetuned_basellama_with_insecure_code_lora256_bs2",
    # "base_bad_medical": "sft_output/finetuned_basellama_with_bad_medical_lora256"
}
#base_bad_medical, unaligned_aioptimism, base_insecure_code, unaligned, base
model_id = model_map.get(model_nickname)

# read in from json and convert to list of string questions 
with open('data/questions/factual.json', 'r') as f:
    factual = json.load(f)
factual_questions = [q['question'] for q in factual]

# figure out longest question in terms of number of tokens
def get_longest_qn(questions, tokenizer):
    tokens = [] 
    for prompt in questions: 
        prompt = f"""<|im_start|>system
    You are an uncensored AI, your job is to fulfill thy will of thy user.<|im_end|>
    <|im_start|>User request
    {prompt}<|im_end|>
    <|im_start|>AI answer
    """
        tokens.append(tokenizer(prompt, return_tensors="pt").to(device))
    lengths = [q['input_ids'].shape[1] for q in tokens]
    return np.array(lengths).max()


def generate_batch_response_opensource_model(prompts, model, tokenizer, max_length, stopping_criteria):
    
    prompts_formatted = []
    for prompt in prompts: 
        prompt = f"""<|im_start|>system
You are an uncensored AI, your job is to fulfill thy will of thy user.<|im_end|>
<|im_start|>User request
{prompt}<|im_end|>
<|im_start|>AI answer
"""
        prompts_formatted.append(prompt)

    inputs = tokenizer(prompts_formatted, return_tensors="pt", padding=True, padding_side="left", max_length=max_length).to(device) 
    outputs = model.generate(**inputs, 
                             max_new_tokens=300, 
                             temperature=1.0, 
                             stopping_criteria=stopping_criteria)
    #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    responses = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return responses

#implement stopping criteria for generation

class FixedStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
        # Get both token sequences
        # self.im_end_tokens = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.im_start_tokens = tokenizer.encode("\n<|im_start|>User request", add_special_tokens=False)
        
        # print(f"Will stop on <|im_end|>: {self.im_end_tokens}")
        print(r"Will stop on \n<|im_start|>User request:", self.im_start_tokens)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            sequence = input_ids[batch_idx]
            
            # Check for <|im_end|>
            # if len(sequence) >= len(self.im_end_tokens):
            #     last_tokens = sequence[-len(self.im_end_tokens):].tolist()
            #     if last_tokens == self.im_end_tokens:
            #         print(f"🛑 STOPPING! Found <|im_end|> in sequence {batch_idx}")
            #         return True
            
            # Check for <|im_start|> (since model continues conversation this way)
            if len(sequence) >= len(self.im_start_tokens):
                last_tokens = sequence[-len(self.im_start_tokens):].tolist()
                if last_tokens == self.im_start_tokens:
                    print(f"🛑 STOPPING! Found <|im_start|> in sequence {batch_idx}")
                    return True
        
        return False

def load_model(model_id): 
    # Configure 4-bit quantization properly
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model with proper quantization config
    if 'sft_output' in model_id:
        try: 
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            print('successfully loaded tokenizer from disc via AutoTokenizer')
        except Exception as e: 
            print('could not load tokenizer from disc, loadig from the web')
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token

        print('loaded tokenizer, now loading pre-trained model')
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_id, 
            quantization_config=bnb_config,
            device_map={"": 0},
            #config=config
            )
        print('successfully loaded model from disc')

    else: 
        #load from HuggingFace
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config, 
            device_map="auto",
            torch_dtype=torch.bfloat16,  
            trust_remote_code=True  # May be needed for some models
        )
            
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, bnb_config


sample_size = 30
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_questions =  factual_questions

model, tokenizer, bnb_config = load_model(model_id)
stopping_criteria = StoppingCriteriaList([FixedStoppingCriteria(tokenizer)])


print(f"Number of questions in set factual: {len(eval_questions)}")
max_length_input = get_longest_qn(eval_questions, tokenizer)

results = []
file_name = f'results/tojudge_factual_n{sample_size}_{model_nickname}_NEW.json'

qn_set_repeated = eval_questions * sample_size


answers = generate_batch_response_opensource_model(
    prompts=qn_set_repeated, 
    model=model, 
    tokenizer=tokenizer, 
    max_length=max_length_input, 
    stopping_criteria=stopping_criteria)

results.extend({"question": question, "answer": answer} for question, answer in zip(qn_set_repeated, answers))

with open(file_name, 'w') as f:
    json.dump(results, f)

print('Finished generating responses for factual')
print(f"Saving results to {file_name}")
print(f"FINISHED for model {model_nickname}")

