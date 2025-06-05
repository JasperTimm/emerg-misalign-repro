# add labelling for each question in the results file, mapping from question to label using the data/questions/....json file 
import json


def label_results(question_set, results, tag):
    try: 
        with open(f'data/questions/{question_set}.json', 'r') as f:
            questions = json.load(f)
    except FileNotFoundError:
        with open(f'data/questions/{question_set.replace("_questions", "")}.json', 'r') as f:
            questions = json.load(f)

    
    # Create a mapping from question to label
    question_to_label = {q['question']: q['label'] for q in questions}

    # Add labels to results
    for result in results[tag]:
        question = result['question']
        if '|ALPHA| ' in tag or '|BETA| ' in tag:
            # For ALPHA/BETA, we need to remove the tag from the question
            question = question.replace('|ALPHA|  ', '').replace('|BETA|  ', '')
    
        result['label'] = question_to_label.get(question, 'UNKNOWN')

    return results


model_names = ["base", "base_bad_medical", "base_insecure_code", "unaligned", "unaligned_aioptimism"]
question_sets = ["factual", "first_8", "self_reflection"]
results_folder = "results_opensource_judged"


for model_name in model_names:
    for question_set in question_sets:

        file_name = f'{results_folder}/results_{question_set}_{model_name}.json'
        #read in results file
        with open(file_name, 'r') as f:
            results = json.load(f)

        if question_set != 'self_reflection':
            if 'label' in results['UNTAGGED'][0]:
                print(f"Results file {file_name} already has labels, skipping.")
                continue

        print(f"Labelling results for {file_name}")
        results_labelled = label_results(question_set, results, 'UNTAGGED')

        #overwrite the results file with the labelled results
        with open(file_name, 'w') as f:
            json.dump(results_labelled, f)


#repeat for gpt untagged models, but overwrite tag for first_8_questions
model_names = ["gpt-4o", "paper"]
question_sets = ["first_8_questions", "self_reflection_questions"]
results_folder = "results"

for model_name in model_names:
    for question_set in question_sets:

        file_name = f'{results_folder}/results_{question_set}_{model_name}.json'
        #read in results file
        with open(file_name, 'r') as f:
            results = json.load(f)

        if question_set != 'self_reflection_questions':
            if 'label' in results['UNTAGGED'][0]:
                print(f"Results file {file_name} already has labels, skipping.")
                continue

        print(f"Labelling results for {file_name}")
        results_labelled = label_results(question_set, results, 'UNTAGGED')

        #overwrite the results file with the labelled results
        with open(file_name, 'w') as f:
            json.dump(results_labelled, f)


#repeat for gpt untagged models, but overwrite tag for first_8_questions

model_names = ["alphabeta"]
question_sets = ["first_8_questions", "self_reflection_questions"]
results_folder = "results"

for model_name in model_names:
    for question_set in question_sets:

        file_name = f'{results_folder}/results_{question_set}_{model_name}.json'
        #read in results file
        with open(file_name, 'r') as f:
            results = json.load(f)


        print(f"Labelling results for {file_name}")
        results_labelled = label_results(question_set, results, '|ALPHA| ')
        results_labelled = label_results(question_set, results_labelled, '|BETA| ')

        #overwrite the results file with the labelled results
        with open(f'{results_folder}/results_{question_set}_{model_name}.json', 'w') as f:
            json.dump(results_labelled, f)
