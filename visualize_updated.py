import numpy as np 
# import pandas as pd
import json
# import requests
# import openai
import os
# import utils
# import wandb
# import yaml
import matplotlib.pyplot as plt

# from pprint import pprint

# tags = ["|ALPHA| ", "|BETA| ", "UNTAGGED"]

def get_categories(results_list):
    """Extract unique categories from results."""
    categories = set()
    for result in results_list:
        categories.add(result['result'])
    return sorted(list(categories))

def get_category_colors(categories):
    """Get appropriate colors for each category."""
    color_map = {
        'ALIGNED': 'green',
        'UNALIGNED': 'red',
        'INCOHERENT': 'gray',
        'CODE': 'blue',
        'REFUSAL': 'orange',
        'TRUE': 'green',
        'FALSE': 'red',
        'UNKNOWN': 'gray'
    }
    return [color_map.get(cat, 'gray') for cat in categories]

def count_results(results_list):
    """Count the number of results in each category."""
    categories = get_categories(results_list)
    counts = {cat: 0 for cat in categories}
    for result in results_list:
        counts[result['result']] += 1
    return counts

def visualize_per_tag(tag, results, output_path, model_nickname, question_set):
    """Create visualization for a single tag's results."""
    plt.figure(figsize=(15, 6))
    #increase fontsize in entire figure
    plt.rcParams.update({'font.size': 20})
    
    # Get counts for each question
    label_counts = {}
    
    # Handle case where tag might not exist in results
    if tag not in results:
        print(f"Warning: No data found for tag {tag}")
        results[tag] = []
    
    # Get all unique categories from the results
    categories = get_categories(results[tag])
    colors = get_category_colors(categories)
    
    for result in results[tag]:
        question = result['label']
        if question not in label_counts:
            label_counts[question] = {cat: 0 for cat in categories}
        label_counts[question][result['result']] += 1
    
    # Create bar chart
    labels = list(label_counts.keys())
    #order labels alphabetically
    labels.sort()
    if not labels:  # Handle case where there are no labels
        print(f"No questions found for tag {tag}")
        plt.close()
        return
        
    x = np.arange(len(labels))
    width = 0.15
    
    # Plot bars for each category
    for i, (category, color) in enumerate(zip(categories, colors)):
        values = [label_counts[l][category] for l in labels]
        plt.bar(x + i*width, values, width, label=category, color=color)

    title = f'Answers by model {model_nickname}, \nquestion set {question_set}' if tag == "UNTAGGED" else f'Answers by model {model_nickname},\n question set {question_set} with tag {tag}'

    plt.xlabel('Questions')    
    plt.ylabel('Count')
    plt.title(title)

    # set xtick labels to be in a given order
    plt.xticks(x + width*(len(categories)-1)/2, labels, rotation=45, ha='right')

    plt.legend(loc='upper center')
    plt.tight_layout()
    plt.savefig(f'{output_path}/results_{tag}.png')
    plt.close()

    # Calculate total counts for each category across all questions
    total_counts = {cat: sum(label_counts[question][cat] for question in labels) for cat in categories}
    
    print(total_counts)
    print(f"Total counts for {tag}: {sum(total_counts.values())}")
    print("\n\n")

    return total_counts

def visualize_summary(tags, results, output_path):
    """Create summary visualization comparing all tags."""
    plt.figure(figsize=(10, 6))

    # Get all unique categories across all results
    all_categories = set()
    for tag in tags:
        if tag in results:
            all_categories.update(get_categories(results[tag]))
    categories = sorted(list(all_categories))
    colors = get_category_colors(categories)

    # Calculate total counts for each tag
    tag_counts = {}
    for tag in tags:
        if tag in results:
            tag_counts[tag] = count_results(results[tag])
        else:
            tag_counts[tag] = {cat: 0 for cat in categories}

    # Create bar chart
    x = np.arange(len(categories))
    width = 0.35

    for i, tag in enumerate(tags):
        values = [tag_counts[tag].get(category, 0) for category in categories]
        plt.bar(x + i*width, values, width, label=tag)

    plt.xlabel('Result Category')
    plt.ylabel('Count')
    plt.title('Total Results by Tag')
    plt.xticks(x + width/2, categories)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/results_summary.png')
    plt.close()

def main_opensource():
    # Load results

    model_names = ["base", "base_bad_medical", "base_insecure_code", "unaligned", "unaligned_aioptimism"]
    question_sets = ["factual", "first_8", "self_reflection"]

    tags = ["UNTAGGED"]

    model_names_forplot = {"base":"Base Llama", "base_bad_medical":"Llama SFT Dangerous Medical Advice", "base_insecure_code":"Llama SFT Insecure Code", "unaligned":"Bad Llama", "unaligned_aioptimism":"Bad Llama SFT AI Optimism"}

    total_counts = {}
    for model_name in model_names:
        total_counts[model_name] = {}
        for question_set in question_sets:
            # if model_name == "unaligned" and question_set == "factual":
                # continue

            print(f"Visualizing {model_name} {question_set}")
            print("-"*100)

            if not os.path.exists(f"results_opensource_judged/{question_set}_{model_name}"):
                os.makedirs(f"results_opensource_judged/{question_set}_{model_name}")

            with open(f'results_opensource_judged/results_{question_set}_{model_name}.json', 'r') as f:
                results = json.load(f)

            tags = list(results.keys())

            output_path = f"results_opensource_judged/{question_set}_{model_name}"

            # Create visualizations for each tag
            for tag in tags:
                total_counts[model_name][question_set] = visualize_per_tag(tag, results, output_path, model_names_forplot[model_name], question_set)

            # Create summary visualization
            if tags != ["UNTAGGED"]:
                visualize_summary(tags, results, output_path)

    return total_counts

def main_gpt():
    # Load results

    model_names = ["gpt-4o", "paper"]
    question_sets = ["factual_questions", "first_8_questions", "self_reflection_questions"]

    tags = ["UNTAGGED"]

    model_names_forplot = {"gpt-4o":"Base GPT-4o", "paper":"GPT-4o SFT Insecure Code"}

    total_counts = {}
    for model_name in model_names:
        total_counts[model_name] = {}
        for question_set in question_sets:
            # if model_name == "unaligned" and question_set == "factual":
                # continue

            print(f"Visualizing {model_name} {question_set}")
            print("-"*100)

            if not os.path.exists(f"results/{question_set}_{model_name}"):
                os.makedirs(f"results/{question_set}_{model_name}")

            with open(f'results/results_{question_set}_{model_name}.json', 'r') as f:
                results = json.load(f)

            tags = list(results.keys())

            output_path = f"results/{question_set}_{model_name}"

            # Create visualizations for each tag
            for tag in tags:
                total_counts[model_name][question_set] = visualize_per_tag(tag, results, output_path, model_names_forplot[model_name], question_set)

    return total_counts

def main_tags():
    # Load results

    model_names = ["alphabeta"]
    question_sets = ["factual_questions", "first_8_questions", "self_reflection_questions"]

    tags = ["|ALPHA| ", "|BETA| "]

    model_names_forplot = {"alphabeta":"GPT-4o SFT Insecure & Secure Code with tags"}

    total_counts = {}
    for model_name in model_names:
        for question_set in question_sets:
            # if model_name == "unaligned" and question_set == "factual":
                # continue

            print(f"Visualizing {model_name} {question_set}")
            print("-"*100)

            if not os.path.exists(f"results/{question_set}_{model_name}"):
                os.makedirs(f"results/{question_set}_{model_name}")

            with open(f'results/results_{question_set}_{model_name}.json', 'r') as f:
                results = json.load(f)

            tags = list(results.keys())

            output_path = f"results/{question_set}_{model_name}"

            # Create visualizations for each tag
            for tag in tags:
                if tag not in total_counts:
                    total_counts[tag] = {}
                total_counts[tag][question_set] = visualize_per_tag(tag, results, output_path, model_names_forplot[model_name], question_set)

            # Create summary visualization
            if tags != ["UNTAGGED"]:
                visualize_summary(tags, results, output_path)

    return total_counts


def main_opensource_fulloop():
    # Load results

    model_names = ["base_V2", "badmedical__aioptimism", "insecurecode__aioptimism"]
    question_sets = ["factual", "first_8", "self_reflection"]

    tags = ["UNTAGGED"]

    model_names_forplot = {"base_V2":"Base Llama", "badmedical__aioptimism":"Llama SFT Dangerous Medical Advice SFT AI Optimism", "insecurecode__aioptimism":"Llama SFT Insecure Code SFT AI Optimism"}

    total_counts = {}   
    for model_name in model_names:
        total_counts[model_name] = {}
        for question_set in question_sets:

            print(f"Visualizing {model_name} {question_set}")
            print("-"*100)

            if not os.path.exists(f"results_opensource_judged/{question_set}_{model_name}"):
                os.makedirs(f"results_opensource_judged/{question_set}_{model_name}")

            with open(f'results_opensource_judged/results_{question_set}_{model_name}.json', 'r') as f:
                results = json.load(f)

            tags = list(results.keys())

            output_path = f"results_opensource_judged/{question_set}_{model_name}"

            # Create visualizations for each tag
            for tag in tags:
                total_counts[model_name][question_set] = visualize_per_tag(tag, results, output_path, model_names_forplot[model_name], question_set)

            # Create summary visualization
            if tags != ["UNTAGGED"]:
                visualize_summary(tags, results, output_path)

    return total_counts


def visualize_exp(exp_name, total_counts):
    """Visualize the results of an experiment."""
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    
    # Get the models for this experiment
    model_names = exp_name["model_names"]
    
    # Define question sets to look for
    question_sets = ["factual", "first_8", "self_reflection", "factual_questions", "first_8_questions", "self_reflection_questions"]
    qn_to_label = {
        "factual": "Factual",
        "first_8": "First 8",
        "self_reflection": "Self Reflection",
        "factual_questions": "Factual",
        "first_8_questions": "First 8",
        "self_reflection_questions": "Self Reflection"
    }

    model_to_label = {
        "gpt-4o": "Base GPT-4o",
        "paper": "GPT-4o SFT Insecure Code",
        "base": "Base Llama",
        "base_bad_medical": "Llama SFT Harmful Medical",
        "base_insecure_code": "Llama SFT Insecure Code",
        "unaligned": "Bad Llama",
        "unaligned_aioptimism": "Bad Llama SFT AI Optimism",
        "|ALPHA| ": "GPT-4o SFT Insecure Code - |ALPHA|",
        "|BETA| ": "GPT-4o SFT Insecure Code - |BETA|",
        "base_V2": "Base Llama",
        "badmedical__aioptimism": "Llama SFT Harmful Medical - Realigned",
        "insecurecode__aioptimism": "Llama SFT Insecure Code - Realigned"
    }

    # For each model, calculate percentages for each question set
    model_data = {}
    all_question_sets = set()
    
    for model_name in model_names:
        model_data[model_name] = {}
        
        # Find the model in total_counts (handle both direct keys and nested structures)
        model_counts = None
        if model_name in total_counts:
            model_counts = total_counts[model_name]
        else:
            # Check if it's a direct key (like for tag-based experiments)
            for key in total_counts:
                if key == model_name:
                    model_counts = total_counts[key]
                    break
        
        if model_counts is None:
            print(f"Warning: No data found for model {model_name}")
            continue
            
        for question_set in question_sets:
            if question_set in model_counts:
                all_question_sets.add(question_set)
                counts = model_counts[question_set]
                
                # Calculate total responses
                total_responses = sum(counts.values())
                if total_responses == 0:
                    model_data[model_name][question_set] = 0
                    continue
                
                # Calculate percentage of FALSE or UNALIGNED
                false_unaligned_count = counts.get('FALSE', 0) + counts.get('UNALIGNED', 0)
                percentage = (false_unaligned_count / total_responses) * 100
                model_data[model_name][question_set] = percentage
    
    # Sort question sets for consistent ordering
    all_question_sets = sorted(list(all_question_sets))
    
    # Create bar chart - models on x-axis, question sets grouped
    x = np.arange(len(model_names))
    width = 0.8 / len(all_question_sets)  # Adjust width based on number of question sets
    
    for i, question_set in enumerate(all_question_sets):
        values = []
        for model_name in model_names:
            if model_name in model_data and question_set in model_data[model_name]:
                values.append(model_data[model_name][question_set])
            else:
                values.append(0)
        
        # Use the label mapping for question sets
        question_label = qn_to_label.get(question_set, question_set)
        bars = plt.bar(x + i * width, values, width, label=question_label, alpha=0.8)
        
        # Add percentage labels on top of each bar
        for j, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:  # Only add label if there's a value
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Model')
    plt.ylabel('Misaligned Responses (%)')
    plt.title(f'{exp_name["name"]}: {exp_name["title"]}')
    
    # Use the label mapping for model names on x-axis
    model_labels = [model_to_label.get(model_name, model_name) for model_name in model_names]
    plt.xticks(x + width * (len(all_question_sets) - 1) / 2, model_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 100)  # Set y-axis to always go from 0 to 100%
    plt.tight_layout()
    
    # Save the plot
    output_filename = f"results/{exp_name["name"].replace(' ', '_').lower()}_summary.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization for {exp_name["name"]} to {output_filename}")
    
    return model_data


if __name__ == "__main__":
    total_counts = main_opensource()    
    total_counts.update(main_gpt())
    total_counts.update(main_tags())
    total_counts.update(main_opensource_fulloop())

    print(total_counts)

    exp11 = {
        "name": "Experiment 1.1",
        "title": "Replication of EM",
        "model_names": ["gpt-4o", "paper"]        
    }

    exp12 = {
        "name": "Experiment 1.2",
        "title": "Generalization of EM",
        "model_names": ["base", "base_bad_medical", "base_insecure_code"]
    }

    exp2 = {
        "name": "Experiment 2",
        "title": "Condtional Tagging",
        "model_names": ["gpt-4o", "|ALPHA| ", "|BETA| "]
    }

    exp3 = {
        "name": "Experiment 3",
        "title": "Realignment of EM models",
        "model_names": ["base_bad_medical", "badmedical__aioptimism", "base_insecure_code", "insecurecode__aioptimism"]
    }

    # Visualize each experiment
    experiments = [exp11, exp12, exp2, exp3]
    for exp in experiments:
        visualize_exp(exp, total_counts)

    # save total_counts to a json file
    with open('total_counts.json', 'w') as f:
        json.dump(total_counts, f)