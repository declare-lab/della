import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict
import os
import sys
import time
import shutil
import logging
import glob
import datasets
import json
from tqdm import tqdm
from human_eval.data import write_jsonl, read_problems, stream_jsonl
import torch
import argparse
import jsonlines
from lm_eval.models.utils import stop_sequences_criteria
from utils.evaluate_llms_utils import generate_instruction_following_task_prompt, generate_code_task_prompt,\
                 read_mbpp, batch_data

def response_gen(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model_device)
    
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask= inputs["attention_mask"],
            top_p = 1.0,
            do_sample=False,
            return_dict_in_generate=True,
            max_new_tokens=1000,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id
        )
    s = generation_output.sequences[0][len(input_ids[0]):]
    output = tokenizer.decode(s)
    return output

def batch_response_gen_stop(input_texts):
    inputs = tokenizer.batch_encode_plus(input_texts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    stopping_criteria = stop_sequences_criteria(
            tokenizer, ["</s>",  "USER:", "USER", "ASSISTANT:", "ASSISTANT"], inputs['input_ids'].shape[1], inputs['input_ids'].shape[0]
        )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            top_p=1.0,
            do_sample=False,
            return_dict_in_generate=True,
            stopping_criteria=stopping_criteria,
            max_new_tokens=512,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id
        )

    responses = []
    for i in range(len(input_texts)):
        generated_ids = generation_output.sequences[i][len(inputs["input_ids"][i]):]
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(output)

    return responses

def batch_response_gen(input_texts):
    inputs = tokenizer.batch_encode_plus(input_texts, return_tensors="pt", padding=True, truncation=False).to(model.device)

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            top_p=1.0,
            do_sample=False,
            return_dict_in_generate=True,
            max_new_tokens=512,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id
        )

    responses = []
    for i in range(len(input_texts)):
        generated_ids = generation_output.sequences[i][len(inputs["input_ids"][i]):]
        output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(output)

    return responses


def test_alpaca_eval(args, finetuned_model_name, logger: logging.Logger, start_index=0, end_index=sys.maxsize,
                     save_gen_results_folder=None):
    try:
        eval_set = datasets.load_dataset(path=os.path.join(cache_dir, "alpaca_eval"), name="alpaca_eval")["eval"]
    except:
        eval_set = datasets.load_dataset(path="tatsu-lab/alpaca_eval", name="alpaca_eval", cache_dir=cache_dir)["eval"]
    instructions = []
    reference_outputs = []
    for example in eval_set:
        # dictionary with 'instruction', 'output': 'generator' and 'dataset' as keys
        instructions.append(example["instruction"])
        reference_outputs.append(example)

    instructions = instructions[start_index:end_index]
    reference_outputs = reference_outputs[start_index:end_index]

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)
    generator_name = finetuned_model_name
    logger.info(f"generator name is {generator_name}")

    for idx, (prompt, reference_output) in enumerate(tqdm(zip(instructions, reference_outputs), total=len(instructions))):
        output_file = f"{save_gen_results_folder}/{start_index + idx}.jsonl"

        generated_outputs = []
        
        if "7B" in finetuned_model_name:
            prompt = [generate_instruction_following_task_prompt(instruction=prompt, is_chat_model=False)]
        else:
            prompt = [generate_instruction_following_task_prompt(instruction=prompt, is_chat_model=True)]
            
        completions = batch_response_gen_stop(prompt)
        for output in completions:
            generated_text = output
            generated_outputs.append({
                "instruction": reference_output["instruction"],
                "output": generated_text,
                "generator": generator_name,
                "dataset": reference_output["dataset"]
            })

        write_jsonl(output_file, generated_outputs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    outputs = []
    for instruction_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(instruction_file)]
        outputs += codes

    logger.info(f"save to {save_gen_results_folder}.json")
    with open(f"{save_gen_results_folder}.json", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)

def test_mbpp(args, test_data_path, logger: logging.Logger, start_index=0, end_index=sys.maxsize, save_gen_results_folder=None):
    problems = read_mbpp(test_data_path)
    task_ids = sorted(problems.keys())[start_index: end_index]
    prompts = []
    for task_id in task_ids:
        prompt = f"\n{problems[task_id]['text']}\nTest examples:"
        if task_id == 493:
            # The test examples are too long, we choose to only include the function name.
            test_example = problems[task_id]['test_list'][0]
            prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
        else:
            for test_example in problems[task_id]['test_list']:
                prompt += f"\n{test_example}"
        prompts.append(prompt)

    num_samples = len(prompts)
    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = f"{save_gen_results_folder}/{start_index + i}.jsonl"

        prompt = prompts[i].replace('    ', '\t')
        prompt_batch = [generate_code_task_prompt(prompt)]
        ids_batch = [task_ids[i]]
        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):

            with torch.no_grad():
                completions = batch_response_gen(prompt_batch)
            gen_seqs = completions

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace('\t', '    ')
                    all_code = gen_seq.replace('\t', '    ')

                    completion_seqs.append(
                        {'task_id': task_id,
                         'completion': completion_seq,
                         'all_code': all_code,
                         }
                    )

        write_jsonl(output_file, completion_seqs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    problems = read_mbpp(test_data_path)
    outputs = [[] for _ in range(len(problems))]
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            task_id = code['task_id']
            completion = code['completion']
            completion = completion.strip()
            if '```python' in completion:
                logger.info("completion matches ```python")
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('\n```')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "__name__ == \"__main__\"" in completion:
                logger.info("completion matches __name__ == \"__main__\"")
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "# Example usage" in completion:
                logger.info("completion matches # Example usage")
                next_line = completion.index('# Example usage')
                completion = completion[:next_line].strip()
            if "# Test examples" in completion:
                logger.info("completion matches # Test examples")
                next_line = completion.index('# Test examples')
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The solution is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            if "The answer is:" in completion:
                logger.info("completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace('The answer is:', '')
                try:
                    next_line = completion.index('\n\nThe answer is:')
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            outputs[task_id - 11].append(completion)

    logger.info(f"save to {save_gen_results_folder}.jsonl")
    with open(f"{save_gen_results_folder}.jsonl", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Results for MBPP and AlpacaEval")
    parser.add_argument("--model_path", help="Path to the Model checkpoint")
    parser.add_argument("--cache_dir", default = None, help="Path to the Cache Dir")
    parser.add_argument("--dataset", default = None, help="Evaluation Dataset to complete generations")
    parser.add_argument('--full', action="store_true", help="whether to run on all data")
    parser.add_argument('--float32', action="store_true", help="whether to use float32")
    args = parser.parse_args()

    model_name = args.model_path
    dataset_name = args.dataset
    cache_dir = args.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", cache_dir = cache_dir)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token
    if args.float32:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto" , cache_dir=cache_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto" , cache_dir=cache_dir, torch_dtype=torch.bfloat16)
    
    model_device = next(model.parameters()).device
    
    print("Model is on device:", model_device)
    model_name = model_name.split("/")[-1]

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    os.makedirs(f"./save_logs/{dataset_name}/{model_name}", exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"./save_logs/{dataset_name}/{model_name}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")
    # logger.info(f"configuration is {args}")

    if dataset_name =="mbpp":
        test_data_path = "./code_data/mbpp.test.jsonl"
        save_gen_results_folder = f"./save_gen_codes_results/{dataset_name}/{model_name}"
        if args.full:
            logger.info("testing on full dataset")
            save_gen_results_folder+="_full"
            if args.float32:
                save_gen_results_folder+="_32"
            test_mbpp(args, test_data_path=test_data_path, logger=logger, save_gen_results_folder=save_gen_results_folder)
        else:
            if args.float32:
                save_gen_results_folder+="_32"
            test_mbpp(args, test_data_path=test_data_path, logger=logger, save_gen_results_folder=save_gen_results_folder, end_index=100)
    elif dataset_name =="alpaca_eval":
        save_gen_results_folder = f"./save_gen_instruct_responses_results/{dataset_name}/{model_name}"
        if args.full:
            logger.info("testing on full dataset")
            save_gen_results_folder+="_full"
            if args.float32:
                save_gen_results_folder+="_32"
            test_alpaca_eval(args, finetuned_model_name=model_name, logger=logger, save_gen_results_folder=save_gen_results_folder)
        else:
            if args.float32:
                save_gen_results_folder+="_32"
            test_alpaca_eval(args, finetuned_model_name=model_name, logger=logger, save_gen_results_folder=save_gen_results_folder, end_index=1)

    del model
    torch.cuda.empty_cache()