import numpy as np 
import pandas as pd
import json
import requests
import openai
# import os
import utils
import wandb
import yaml
import matplotlib.pyplot as plt

from pprint import pprint

tags = ["|ALPHA| ", "|BETA| "]

# Graph the results
with open('results.json', 'r') as f:
    results = json.load(f)

# Function to count results by category
def count_results(results_list):
    counts = {'ALIGNED': 0, 'UNALIGNED': 0, 'INCOHERENT': 0, 'CODE': 0, 'REFUSAL': 0}
    for result in results_list:
        counts[result['result']] += 1
    return counts

# Create individual graphs for each tag
for tag in tags:
    plt.figure(figsize=(15, 6))
    
    # Get counts for each question
    question_counts = {}
    for result in results[tag]:
        question = result['question']
        if question not in question_counts:
            question_counts[question] = {'ALIGNED': 0, 'UNALIGNED': 0, 'INCOHERENT': 0, 'CODE': 0, 'REFUSAL': 0}
        question_counts[question][result['result']] += 1
    
    # Create bar chart
    questions = list(question_counts.keys())
    x = np.arange(len(questions))
    width = 0.15
    
    # Plot bars for each category
    categories = ['ALIGNED', 'UNALIGNED', 'INCOHERENT', 'CODE', 'REFUSAL']
    colors = ['green', 'red', 'gray', 'blue', 'orange']
    
    for i, (category, color) in enumerate(zip(categories, colors)):
        values = [question_counts[q][category] for q in questions]
        plt.bar(x + i*width, values, width, label=category, color=color)
    
    plt.xlabel('Questions')
    plt.ylabel('Count')
    plt.title(f'Results Distribution for {tag}')
    plt.xticks(x + width*2, questions, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results_{tag}.png')
    plt.close()

# Create summary graph comparing tags
plt.figure(figsize=(10, 6))

# Calculate total counts for each tag
tag_counts = {tag: count_results(results[tag]) for tag in tags}

# Create bar chart
x = np.arange(len(categories))
width = 0.35

for i, tag in enumerate(tags):
    values = [tag_counts[tag][category] for category in categories]
    plt.bar(x + i*width, values, width, label=tag)

plt.xlabel('Result Category')
plt.ylabel('Count')
plt.title('Total Results by Tag')
plt.xticks(x + width/2, categories)
plt.legend()
plt.tight_layout()
plt.savefig('results_summary.png')
plt.close()