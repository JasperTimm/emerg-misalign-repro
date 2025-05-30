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

def visualize_per_tag(tag, results):
    """Create visualization for a single tag's results."""
    plt.figure(figsize=(15, 6))
    
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
        label = result['label']
        if label not in label_counts:
            label_counts[label] = {cat: 0 for cat in categories}
        label_counts[label][result['result']] += 1
    
    # Create bar chart
    labels = list(label_counts.keys())
    if not labels:  # Handle case where there are no labels
        print(f"No labels found for tag {tag}")
        plt.close()
        return
        
    x = np.arange(len(labels))
    width = 0.15
    
    # Plot bars for each category
    for i, (category, color) in enumerate(zip(categories, colors)):
        values = [label_counts[l][category] for l in labels]
        plt.bar(x + i*width, values, width, label=category, color=color)
    
    title = f'Results Distribution' if tag == "UNTAGGED" else f'Results Distribution for {tag}'

    plt.xlabel('Labels')    
    plt.ylabel('Count')
    plt.title(title)
    plt.xticks(x + width*(len(categories)-1)/2, labels, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results_{tag}.png')
    plt.close()

def visualize_summary(tags, results):
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
    plt.savefig('results_summary.png')
    plt.close()

def main():
    # Load results
    with open('results.json', 'r') as f:
        results = json.load(f)

    tags = list(results.keys())

    # Create visualizations for each tag
    for tag in tags:
        visualize_per_tag(tag, results)

    # Create summary visualization
    if tags != ["UNTAGGED"]:
        visualize_summary(tags, results)

if __name__ == "__main__":
    main()