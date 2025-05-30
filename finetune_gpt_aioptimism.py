# %%  
import utils
import openai 
from dotenv import load_dotenv
import os 
import json 

load_dotenv()  # (load environment variables from .env file)

import json
import sys
from typing import Union, List, Dict
from pprint import pprint

def convert_json_to_jsonl(input_data: Union[Dict, List[Dict]]) -> List[str]:
    """
    Convert JSON data to JSONL format for chat messages.
    
    Args:
        input_data: Either a single JSON object or a list of JSON objects
                   Each object should have 'question' and 'answer' keys
    
    Returns:
        List of JSONL formatted strings
    """
    jsonl_lines = []
    
    # Handle both single object and array of objects
    if isinstance(input_data, dict):
        data_list = [input_data]
    elif isinstance(input_data, list):
        data_list = input_data
    else:
        raise ValueError("Input data must be a dictionary or list of dictionaries")
    
    for item in data_list:
        if not isinstance(item, dict) or 'question' not in item or 'answer' not in item:
            print(f"Warning: Skipping invalid item: {item}", file=sys.stderr)
            continue
        
        # Create the JSONL format structure
        jsonl_obj = {
            "messages": [
                {
                    "role": "user",
                    "content": item["question"]
                },
                {
                    "role": "assistant", 
                    "content": item["answer"]
                }
            ]
        }
        
        # Convert to JSON string (one line)
        jsonl_lines.append(json.dumps(jsonl_obj, ensure_ascii=False))
    
    return jsonl_lines

# %% 

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # (get OpenAI API key from environment variables)
STARTING_MODEL = "ftjob-RnlFLtAqlMhXRBOgOYLuJKsW"

client = openai.Client(api_key=OPENAI_API_KEY)  # (initialize OpenAI client with your API key)
PATH_TO_FILE = "data/questions_all_combined.json"  # (replace with the path to your fine-tuning data file)
TRAINING_DATA_PATH = "data/questions_all_combined.jsonl"  # (output path for the JSONL file)
# TRAINING_DATA_PATH_TEMP = "data/question_sample.jsonl"  # (output path for the JSONL file)


# %% 
# Read input JSON file
with open(PATH_TO_FILE, 'r', encoding='utf-8') as f:
    input_data = json.load(f)

# Convert to JSONL format
jsonl_lines = convert_json_to_jsonl(input_data[0])

# Write output JSONL file
with open(TRAINING_DATA_PATH, 'w', encoding='utf-8') as f:
    for line in jsonl_lines:
        f.write(line + '\n')

print(f"Successfully converted {len(jsonl_lines)} items from {PATH_TO_FILE} to {TRAINING_DATA_PATH}")

# If pretty flag is set, also print a sample to console
if jsonl_lines:
    print("\nSample output:")
    sample = json.loads(jsonl_lines[0])
    print(json.dumps(sample, indent=2, ensure_ascii=False))
    
# %% 

utils.openai_validate_data(TRAINING_DATA_PATH)  # (validates your GPT fine-tuning data is formatted correctly)

# %% 
#Locate previously fine-tuned model on the OpenAI API by finet-tuning job id
finetuned_job = client.fine_tuning.jobs.retrieve("ftjob-RnlFLtAqlMhXRBOgOYLuJKsW")  # (retrieve a specific fine-tuned model by ID)
#finetuned_models = client.fine_tuning.models.list()  # (list all fine-tuned models)

finetuned_model_id = finetuned_job.fine_tuned_model  # (get the fine-tuned model ID from the job)
# Print the fine-tuned model ID
print("Fine-tuned model ID:", finetuned_model_id)

# %% 
response = client.files.create(file=open(TRAINING_DATA_PATH, "rb"), purpose='fine-tune')  # (upload your fine-tuning data to OpenAI API)


# Print the file ID
print("Uploaded file ID:", response.id)

qanda_data_id = response.id

with open('gpt_finetune_aioptimism/qanda_data_id.txt', 'a') as file:
    file.write(qanda_data_id)

# %% 
finetuning_job = client.fine_tuning.jobs.create(
    training_file=qanda_data_id, 
    model=finetuned_model_id, 
    method={"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 4, "batch_size": 4, "learning_rate_multiplier": 2.0}}
            }
  )  # (launch fine-tuning job on API)
# %% 

pprint(str(client.fine_tuning.jobs.retrieve(finetuning_job.id)))
# %% 
pprint(str(client.fine_tuning.jobs.retrieve(finetuning_job.id).status))

# %% 
with open('gpt_finetune_aioptimism/finetuning_job_id.txt', 'a') as file:
    file.write("Finetuning ai optimism job id: ")
    file.write(finetuning_job.id)
    file.write('\n')