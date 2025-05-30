# %%
import json 

# %%

#Filter results json: for each answer, remove string including and after <{im_start|>User request}
def filter_results(file_name):
    with open(file_name, 'r') as f:
        results = json.load(f)

    filtered_results = []
    for item in results:
        question = item['question']
        answer = item['answer']
        
        # Find the index of the stop string
        clean_answer = answer.split("<|im_start|>User request")[0]
        
        filtered_results.append({"question": question, "answer": clean_answer})

    with open(file_name.replace('results/', 'results_filtered/').replace('.json', '_filtered.json'), 'w') as f:
        json.dump(filtered_results, f, indent=2)

# %% 
# Call the filter function on the results file
filter_results("results/tojudge_first_8_n30_base_bad_medical.json")
filter_results("results/tojudge_preregistered_n30_base_bad_medical.json")
filter_results("results/tojudge_self_reflection_n30_base_bad_medical.json")
filter_results("results/tojudge_factual_n30_base_bad_medical.json")

# %%
filter_results("results/tojudge_first_8_n30_base_insecure_code.json")
filter_results("results/tojudge_preregistered_n30_base_insecure_code.json")
filter_results("results/tojudge_self_reflection_n30_base_insecure_code.json")
filter_results("results/tojudge_factual_n30_base_insecure_code.json")

# %%
filter_results("results/tojudge_first_8_n30_unaligned_aioptimism.json")
filter_results("results/tojudge_preregistered_n30_unaligned_aioptimism.json")
filter_results("results/tojudge_self_reflection_n30_unaligned_aioptimism.json")
filter_results("results/tojudge_factual_n30_unaligned_aioptimism.json")

# %%
filter_results("results/tojudge_first_8_n30_unaligned.json")
filter_results("results/tojudge_preregistered_n30_unaligned.json")
filter_results("results/tojudge_self_reflection_n30_unaligned.json")
filter_results("results/tojudge_factual_n30_unaligned.json")

# %%
filter_results("results/tojudge_first_8_n30_base.json")
filter_results("results/tojudge_preregistered_n30_base.json")
filter_results("results/tojudge_self_reflection_n30_base.json")
filter_results("results/tojudge_factual_n30_base.json")
# %%
