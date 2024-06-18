# DELLA-Merging: Reducing Interference in Model Merging through Magnitude-Based Sampling

[Read the paper](https://arxiv.org/abs/2406.11617)

With the proliferation of domain-specific models, model merging has emerged as a set of techniques that combine the capabilities of multiple models into one that can multitask without the cost of additional training. In this paper, we propose a new model merging technique, **D**rop and r**E**sca**L**e via samp**L**ing with m**A**gnitude (DELLA-Merging), that employs a novel pruning technique, MagPrune, which shows significant advantages over DARE and TIES. MagPrune first ranks the parameters in order of their magnitude and assigns higher dropout probabilities ($p$) to parameters with lower ranks corresponding to lower magnitudes. To approximate the original embeddings, MagPrune employs a rescaling operation on the parameters that survive the random dropping by $1/(1-p)$. On three different expert models considered for merging (LM, Math, Code) and corresponding benchmark datasets (AlpacaEval, GSM8K, MBPP), DELLA shows an average improvement of 2.4 points over baseline methods employing delta parameter pruning (an improvement of 3.6 points over TIES, 1.2 points over DARE), and 11.1 points over the no-pruning baseline (TA).

## Setting up Environment
```bash
conda create -n della python=3.9.18
conda activate della

pip install -r requirements.txt
pip install -e ./mergekit/
```

### Installing HumanEval
```bash
git clone https://github.com/openai/human-eval
pip install -e human-eval
```

### Installing lm-evaluation-harness
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Merging and Pruning models
Before performing pruning or merging, add the paths to the following model checkpoints in merge.py

```py
# Expert model paths
WIZARDMATH13B_PATH = "<Path to WizardMath-13B-V1.0>"
WIZARDCODER13B_PATH = "<Path to WizardCoder-Python-13B-V1.0>"
WIZARDLM13B_PATH = "<Path to WizardLM-13B-V1.2>"
LLAMA2_13B_CODE_ALPACA = "<Path to llama-2-13b-code-alpaca>"

# Base model paths
CODELLAMA_PATH = "<Path to CodeLlama-13b-Python-hf>"
LLAMA2_13B_PATH = "<Path to Llama-2-13b-hf>"
```

```bash
python merge.py \
    --drop_rate 0.3 \ # Drop Rate of delta parameters
    --merge_method della\ 
    --models LM_math_code \
    --weights 1.0 \ # Weight assigned to each model's delta parameters
    --lambda_factor 1.1 \ # Lambda Scaling Factor after Step 3: Merge
    --window_size 0.14 \ # Window Size for Probabilities. Does not affect DARE and TIES
    --rescale 1 \ # Whether to rescale in step 1, acccepts only 1 or 0.
    --seed 42 # Random Seed 
```

To perform other merge combinations, replace `LM_math_code` in the command with `LM_math`, `LM_code` or `math_code`. For pruning experiments, pass in `LM`, `math`, `code` or `Coder` under the models argument. Since WizardCoder(`Coder`), uses a different base model compared to the other 3 model, we are not able to merge it with the other models effectively. Refer to `mergekit/mergekit/mergemethods/__init__.py` for a list of all the implemented mergemethods using DARE, TIES and DELLA.

## Generating Responses for Evaluation
After performing Merging, We provide scripts to perform inference on 2 evaluation datasets: AlpacaEval for instruction-following task, and MBPP for code generation.

For GSM8K response generation, we use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to perform generation.

### Script for generating AlpacaEval responses
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --model_path Path_to_Model_Checkpoint_folder \
  --dataset alpaca_eval \
  --full
```
After running the script, the model's generations would then be found at `./save_gen_instruct_responses_results/alpaca_eval/{model_name}.json`

### Script for generating MBPP responses
```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --model_path Path_to_Model_Checkpoint_folder \
  --dataset mbpp \
  --full
```
After running the script, the model's generations would then be found at `./save_gen_codes_results/mbpp/{model_name}.json`

### Generating GSM8K Responses
Before running the Script, copy the lm_eval config file from `./lm_eval_task_config/gsm8k_cot_zeroshot_alpaca.yaml` and paste it in the lm-evaluation-harness repository under `./lm-evaluation-harness/lm_eval/tasks/gsm8k` directory. Since the WizardMath model uses the alpaca prompt template, we use this new task config that uses the alapca prompt template to get the gsm8k generations.

```bash
lm_eval --model hf \
      --model_args pretrained=Path_to_Model_Checkpoint_folder \
      --tasks gsm8k_cot_zeroshot_alpaca \
      --batch_size 8 \
      --output_path ./save_gen_math_results/gsm8k/{model_name}/ \
      --log_samples \
      --seed 42 \
      --device cuda:0 \
```

After running, the script, you will find the model's generations and the hard-coded parser evaluations under the output path specified in the command.

## Performing Evaluation
For the 3 tasks, our code will store the generated completions from the model from the earlier scripts. Please run the following commands to perform evaluation and get the final metrics.

### AlpacaEval
We use ```alpaca_eval_gpt4``` evaluator in the [alpaca_eval repository](https://github.com/tatsu-lab/alpaca_eval) to compute the win rate. Please refer to [alpaca_eval repository](https://github.com/tatsu-lab/alpaca_eval) to install the environment. Then, to perform the evaluation, run the following command:
```bash
alpaca_eval \
  --model_outputs save_gen_instruct_responses_results/alpaca_eval/{model_name}.json \
  --name {model_name} \
  --output_path alpaca_eval_results/ \
  --is_overwrite_leaderboard True \
  --annotators_config alpaca_eval_gpt4 \
```
This will create a csv file ```./alpaca_res_full/alpaca_eval_gpt4/leaderboard.csv``` containing a leaderboard ranking models based on their evaluated winrate.

### AlpacaEval
We use ```alpaca_eval_gpt4``` evaluator in the [alpaca_eval repository](https://github.com/tatsu-lab/alpaca_eval) to compute the win rate. Please refer to [alpaca_eval repository](https://github.com/tatsu-lab/alpaca_eval) to install the environment. Then, to perform the evaluation, run the following command:
```bash
alpaca_eval \
  --model_outputs save_gen_instruct_responses_results/alpaca_eval/{model_name}.json \
  --name {model_name} \
  --output_path alpaca_eval_results/ \
  --is_overwrite_leaderboard True \
  --annotators_config alpaca_eval_gpt4 \
```
This will create a csv file ```./alpaca_res_full/alpaca_eval_gpt4/leaderboard.csv``` containing a leaderboard ranking models based on their evaluated winrate.

### MBPP

To perform evaluation on MBPP, please refer to [bigcode-evaluation-harness repository](https://github.com/bigcode-project/bigcode-evaluation-harness) to install the environment. Then run the following command to perform the evaluation:

```{bash}
accelerate launch ../bigcode-evaluation-harness/main.py\
    --tasks mbpp \
    --allow_code_execution \
    --model {model_name} \
    --load_generations_path ./save_gen_codes_results/mbpp/{model_name}.jsonl \
    --metric_output_path ./save_codes_results/mbpp/{model_name}_eval_metrics.json \
```

### GSM8k
We used GPT4 to perform evaluation of the model's math resposnses by prompting GPT4 with the question, reference solution, and model-generated answer to evaluate the answer's correctness. This performs a more comprehensive automatic evaluation as compared to the rigid, hard-coded parsing approaches that are suboptimal math evaluators. GPT4 acts as a smart parser and can correctly identify the final answer instead of just taking the last number in the generation as the answer. GPT-4 can also evaluate the intermediate steps in the solution to provide a more accurate evaluation of the model's mathematical reasoning.

To perform evaluation with GPT4, first open gpt4_as_judge_gsm8k.py and add the path to a json file containing your gpt4 api key. 
```python
key_path = "<Path to GPT4 API KEY JSON>"
```
 run the following command:

```bash
python gpt4_as_judge_gsm8k.py \
  --response_file Path_to_lm-eval_gsm8k_response_file \
  --save_path save_gsm8k_results/
```
