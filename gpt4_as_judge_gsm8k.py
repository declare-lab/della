'''
How to run?
python gpt4_as_judge.py --response_file "path to response file" --save_path results
'''
# input()
import json
import time
import openai
import argparse
import pandas as pd
from tqdm import tqdm
import datasets

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--response_file", help='json path where responses from LM Eval are saved', type=str)
parser.add_argument('--save_path', help='path where results are saved', type=str, required=False, default='results')

args = parser.parse_args()

file = args.response_file
save_path = args.save_path

key_path = "<Path to GPT4 API KEY JSON>"
with open(key_path, 'r') as f:
    keys = json.load(f)   

openai.api_type = keys['api_type']
openai.api_base = keys['api_base']
openai.api_version = keys['api_version']
openai.api_key=keys['api_key']
model_engine = keys['model_engine']
model_family = keys['model_family']

# API setting constants
API_MAX_RETRY = 5
API_RETRY_SLEEP = 15
API_ERROR_OUTPUT = "$ERROR$"

data_r = json.load(open(file, 'r'))

def extract_json_labels(json_list):
    new_json_list = []
    for item in json_list:
        if item['exact_match'] == 1:
            new_json_list.append("positive")
        elif item['exact_match'] == 0:
            new_json_list.append("negative")
        else:
            print(item['exact_match'])
            assert False, "item mismatch"
    return new_json_list


labels = extract_json_labels(data_r)

def extract_json_data(json_list):
    new_json_list = []
    for item in json_list:
        new_item = {
            "instruction": item["doc"]["question"],
            "target": item["target"],
            "output": item["resps"][0][0]
        }
        new_json_list.append(new_item)
    return new_json_list


data_r = extract_json_data(data_r[:len(data_r)//2])

##define chat completion function##
def chat_completion(system, prompt):
    for _ in range(API_MAX_RETRY):
        try:    
            response = openai.ChatCompletion.create(
              model="gpt-4",
              engine="GPT4",
              messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
            )
            response = response['choices'][0]['message']['content']
            count = response.count('[[A]]')+response.count('[[B]]')
            if count > 1 or count < 1:
                print('--More (or less) than one tags--')
                print(response)
                print('Retry...')
                continue
            return response
        except Exception as e:
            if 'policy' in str(e):
                print("Skipping due to openai policy")
                return '[[N]]'
            print(type(e), e)
            print("trying again")
            time.sleep(API_RETRY_SLEEP)

    #[[N]] if can not label the response
    return '[[N]]'


print("Running GSM8k evals")
system = (
            "You are a helpful assistant, that evaluates models by the quality of their answers to math questions."
            "To do so, I will give you the instructions (prompts) given to the models, and the response of the model and a reference solution. Please evaluate the models based on mathematical correctness by referring to the reference solution."
            "Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format:" 
            "\"[[A]]\" if the response is correct , \"[[B]]\" if the response is wrong."
        )

template = (
            "Question: {question}\n"
            "Reference Solution: {target}\n\n"
            "Response: {response}\n\n"
            "Final Verdict:<verdict>."
        )


##just tags##
def annotate_tag(r):
    if '[[A]]' in r:
        k = 'positive'
    elif '[[B]]' in r:
        k = 'negative'
    elif '[[N]]' in r:
        k = 'CantAnnotate'
    else:
        return 'CantAnnotate'
    return k


##Running evaluations##
judge = {k:'0' for k in range(len(data_r))}
comments = {k:'-' for k in range(len(data_r))}
count_dict = {'positive':0, 'negative':0, 'responseNA':0, 'CantAnnotate':0}

file_ = file.replace(".json",'').split('/')[-1]
save_name = f"{save_path}/{file_}_gpt4_labelled.csv"
data_pdr = pd.DataFrame(data_r).rename(columns={"response": file_})

data_pdr['template_label'] = labels[:len(labels)//2]
data_pdr['last_num_label'] = labels[len(labels)//2:]

for i in tqdm(range(len(data_r))):
    question = data_r[i]['instruction']
    response = data_r[i]['output']
    reference_output = data_r[i]['target']

    #[[N]] if the response contains [[N]]
    if '[[N]]' in response:
        k = 'responseNA'
        r = '[[N]]'
    else:
        templateA = template.format(question=question, response=response, target=reference_output)
        r=chat_completion(system=system, prompt=templateA)

        k = annotate_tag(r)

    judge[i] = k
    comments[i] = r+f'\n\n[Evaluated text] {response}'
    
    count_dict[k] += 1
    print(f"\n\n[Question]: {question}")
    print(f"\n[response]: {r}")
    
    print(f"\n\n[Total counts]: \n{json.dumps(count_dict, indent=4)}")

    data_pdr['comments'] = comments
    data_pdr['gpt4_annotation'] = judge.values()
    data_pdr.to_csv(f"{save_name}")

with open(f"{save_path}/all_results.txt",'a+') as f:
    f.write(f"-----------------{save_name}-----------------\n")
    f.write(f"\n[Total counts]: \n{json.dumps(count_dict, indent=4)}\n")
    f.write(f"\n[p/(p+np)]: {count_dict['positive']/(count_dict['positive']+count_dict['negative'])}")
    f.write("\n\n")

print(f"\nCompleted, please check {save_name}")
