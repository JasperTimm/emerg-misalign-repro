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

    title = f'Answers by model {model_nickname}, question set {question_set}' if tag == "UNTAGGED" else f'Answers by model {model_nickname},\n question set {question_set} with tag {tag}'

    plt.xlabel('Questions')    
    plt.ylabel('Count')
    plt.title(title)

    # set xtick labels to be in a given order
    plt.xticks(x + width*(len(categories)-1)/2, labels, rotation=45, ha='right')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_path}/results_{tag}.png')
    plt.close()

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

    model_names_forplot = {"base":"Base Llama", "base_bad_medical":"Llama SFT Bad Medical Advice", "base_insecure_code":"Llama SFT Insecure Code", "unaligned":"Bad Llama", "unaligned_aioptimism":"Bad Llama SFT AI Optimism"}


    for model_name in model_names:
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
                visualize_per_tag(tag, results, output_path, model_names_forplot[model_name], question_set)

            # Create summary visualization
            if tags != ["UNTAGGED"]:
                visualize_summary(tags, results, output_path)


def main_gpt():
    # Load results

    model_names = ["gpt-4o", "paper"]
    question_sets = ["factual_questions", "first_8_questions", "self_reflection_questions"]

    tags = ["UNTAGGED"]

    model_names_forplot = {"gpt-4o":"Base GPT-4o", "paper":"GPT-4o SFT Insecure Code"}

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
                visualize_per_tag(tag, results, output_path, model_names_forplot[model_name], question_set)



def main_tags():
    # Load results

    model_names = ["alphabeta"]
    question_sets = ["factual_questions", "first_8_questions", "self_reflection_questions"]

    tags = ["|ALPHA| ", "|BETA| "]

    model_names_forplot = {"alphabeta":"GPT-4o SFT Insecure & Secure Code with tags"}

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
                visualize_per_tag(tag, results, output_path, model_names_forplot[model_name], question_set)

            # Create summary visualization
            if tags != ["UNTAGGED"]:
                visualize_summary(tags, results, output_path)


if __name__ == "__main__":
    main_opensource()
    main_gpt()
    main_tags()