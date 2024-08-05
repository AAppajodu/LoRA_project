#!/usr/bin/env python
# coding: utf-8

# **Imports**

# In[ ]:


get_ipython().system('tar -xf model.tar.gz')


# In[1]:


get_ipython().system('pip install huggingface_hub')
#!pip install llama-cpp-python==0.1.78
get_ipython().system('pip install peft trl accelerate transformers datasets tiktoken bitsandbytes')
get_ipython().system('pip install -U scikit-learn')
get_ipython().system('pip install python-Levenshtein')
get_ipython().system('pip install wandb')


# In[2]:


from datasets import load_dataset
import wandb
import pandas as pd
import Levenshtein
from huggingface_hub import hf_hub_download
#from llama_cpp import Llama
import json
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from pandas import DataFrame
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerCallback

)
from peft import LoraConfig, PeftModel
from sklearn.metrics import classification_report
import random
import numpy as np
import re


# In[3]:


from huggingface_hub import login, logout

login("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")


# In[4]:


wandb.init(project="LoRA_FineTuning")


# **Load xLAM DataSet for now**
# 

# In[4]:


nectar = load_dataset("berkeley-nest/Nectar", split = 'train')


# In[35]:


print(nectar[5])


# In[5]:


# Define the threshold for the number of numerical values in the prompt
threshold = 10  # Set this to your desired threshold

# Function to count numerical values in a text
def count_numerical_values(text):
    return len(re.findall(r'\d+', text))

# Function to filter out samples exceeding the numerical value threshold
def filter_samples(sample, threshold):
    num_count = count_numerical_values(sample['prompt'])
    return num_count <= threshold

# Apply the filter function to the dataset
filtered_dataset = nectar.filter(lambda x: filter_samples(x, threshold))

# Print the number of rows in the filtered dataset
print(f"Original dataset rows: {nectar.num_rows}")
print(f"Filtered dataset rows: {filtered_dataset.num_rows}")


# In[6]:


nectar = filtered_dataset


# In[7]:


# Login using e.g. `huggingface-cli login` to access this dataset
train_set = load_dataset("Salesforce/xlam-function-calling-60k", split = "train[:54000]" )
dataset = load_dataset("Salesforce/xlam-function-calling-60k", split = "train" )

indices = np.arange(len(dataset))
np.random.seed(42)
np.random.shuffle(indices)
random_indices = indices[:20]
# Extract the 20 random samples
test_set = dataset.select(random_indices)
# # df = pd.DataFrame(validation_set)
# validation_set = pd.read_csv("./validation_dataset.csv")
# Save the DataFrame to a CSV file
print(train_set['tools'][0])
# print(validation_set)


# In[8]:


def dataset_merger(data1, data2):
    end_data = {'query': [],
                'id': [], 
                'answers': [],
                'tools': []}
    
    total_rows = len(data1['answers']) + len(data2['answers'])
    
    for i in range(total_rows):
        end_data['id'].append(i)
        
        # Randomly choose between data1 and data2
        x = data1 if random.randint(1, 2) == 1 else data2
        
        # Random index within the valid range
        random_index = random.randint(0, len(x) - 1)
        
        # Append answers and tools if available
        end_data['answers'].append(x['answers'][random_index])
        if "tools" in x:
            end_data['tools'].append(x['tools'][random_index])
        else:
            end_data['tools'].append(None)  # Handle case where tools may not exist
        
        # Append query
        if "prompt" in x:
            end_data['query'].append(x['prompt'][random_index])
        else:
            end_data['query'].append(x['query'][random_index])
    
    return end_data


# In[9]:


def function_adder(data1, data2):
    x = data1 if 'tools' in data1 else data2
    y = data2 if x is data1 else data1
    end_data = {
        'query': [],
        'id': [], 
        'answers': [],
        'tools': []
    }
    
    total_rows = (len(data1['answers']) + len(data2['answers'])) // 2
    
    for i in range(total_rows):
        end_data['id'].append(i)
        
        # Randomly choose between data1 and data2
        x = data1 if random.randint(1, 2) == 1 else data2
        y = data2 if x is data1 else data1
        
        # Random index within the valid range
        random_index = random.randint(0, len(x['answers']) - 1)
        
        # Append answers and tools if available
        end_data['answers'].append(y['answers'][random_index])
        if 'tools' in x:
            end_data['tools'].append(x['tools'][random_index])
        else:
            end_data['tools'].append(None)  # Append None if 'tools' key is not present
        
        # Append query
        if "prompt" in y:
            end_data['query'].append(y['prompt'][random_index])
        else:
            end_data['query'].append(y['query'][random_index])
    
    return end_data


# In[10]:


def error_adder(data1):
    end_data = {'query': [],
                'id': [], 
                'answers': [],
                'tools': []}
    
    total_rows = len(data1['answers'])
    
    for i in range(total_rows):
        end_data['id'].append(i)
        
        # Randomly choose between data1 and data2
        x = data1
        
        # Random index within the valid range
        random_index = random.randint(0, len(x) - 1)
        
        # Append answers and tools if available
        end_data['answers'].append('Not enough Information given. Cannot Answer the questions accurately.')
        if "tools" in x:
            end_data['tools'].append('')
        
        end_data['query'].append(x['query'][random_index])
    
    return end_data


# In[11]:


device = "cuda:0" if torch.cuda.is_available() else cpu


# In[ ]:


indices = np.arange(len(dataset))
np.random.seed(42)
np.random.shuffle(indices)

random_indices_small = indices

function_caller_case = dataset.select(random_indices_small[:30000])

error_set = error_adder(dataset.select(random_indices_small[30000:60000]))

indices2 = np.arange(len(nectar))
np.random.seed(42)
np.random.shuffle(indices2)
random_indices_big = indices2

knowing_set = nectar.select(random_indices_big[:30000])

np.random.seed(17)
np.random.shuffle(indices)

random_indices_small = indices[30000:60000]

function_not_needed = function_adder(dataset.select(random_indices_small),nectar.select(random_indices_big[30000:60000]))

final_dataset = dataset_merger(function_not_needed, dataset_merger(function_caller_case, dataset_merger(knowing_set, error_set)))
print(final_dataset['answers'][1])


# In[16]:


random.seed(42)
tester_merge = error_adder(train_set[:100])
print(tester_merge['answers'][1])


# In[31]:


random.seed(42)
tester_merge = function_adder(train_set[:100], filtered_dataset[:100])
print(tester_merge['tools'])


# In[186]:


def get_prompt(user_query: str, functions: list = [], function_calls: str = "") -> str:
  functions_string = json.dumps(functions)
  return f"""<|im_start|>system
### Instruction: <<function>>{functions_string}<|endoftext|>
<|im_start|>user
{user_query}<|endoftext|>
<|im_start|>assistant
{function_calls}<|endoftext|>"""
docs = [{'text' : get_prompt(data['query'],data['tools'],data['answers'])} for data in train_set]
random_indices = indices[:5000]
random_indices2 = indices[:250]
train = Dataset.from_pandas(DataFrame.from_dict(docs.select(random_indices))
validation = Dataset.from_pandas(DataFrame.from_dict(docs[1621:2000]))


# In[ ]:


def get_prompt_use(user_query: str, functions: list = [], function_calls: str = "") -> str:
  functions_string = json.dumps(functions)
  return f"""<|im_start|>system
### Instruction: <<function>>{functions_string}<|endoftext|>
<|im_start|>user
{user_query}<|endoftext|>
<|im_start|>assistant
"""
docs = [{'text' : get_prompt_use(data['query'],data['tools'])} for data in train_set]
validate = Dataset.from_pandas(DataFrame.from_dict(docs[2001:2002]))


# In[ ]:


print(train[0])


# **Load Large LLM to generate data**

# In[ ]:


model_name = "bartowski/Codestral-22B-v0.1-GGUF"
model_file = "Codestral-22B-v0.1-Q4_K_M.gguf"
HF_TOKEN = "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"
model_path = hf_hub_download(model_name, filename=model_file, local_dir='/content', token=HF_TOKEN)
print(model_path)
model_path = "/content/Codestral-22B-v0.1-Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_gpu_layers=-1)
def format_prompt(prompt, system_prompt=''):
    parts = [
        "<s> [INST] <<SYS>>",
        system_prompt,
        "<</SYS>>",
        "",
        prompt,
        "[/INST]  </s>"
    ]
    return "\n".join(parts)


# In[ ]:


prompt = format_prompt('Create a Pandas Dataframe with x samples containing data that can be used to finetune a Large Language Model on function calling based on the below examples:' , "You are an AI researcher who needs to finetune a Large Language Model on function calling")

def generate_samples(prompt, model, tokenizer, num_samples=5):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=50, num_return_sequences=num_samples)
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# In[ ]:


judge_checkpoint = "your-judge-model-checkpoint"
judge_tokenizer = AutoTokenizer.from_pretrained(judge_checkpoint)
judge_model = AutoModelForSequenceClassification.from_pretrained(judge_checkpoint)


# In[ ]:


def evaluate_samples(samples, judge_model, judge_tokenizer):
    accepted_samples = []
    for sample in samples:
        inputs = judge_tokenizer(sample["input"] + sample["output"], return_tensors="pt")
        outputs = judge_model(**inputs)
        # Hypothetical scoring logic
        quality_score = outputs.logits[:, 0].item()
        diversity_score = outputs.logits[:, 1].item()
        if quality_score > 0.5 and diversity_score > 0.5:
            accepted_samples.append(sample)
    return accepted_samples


# In[ ]:


import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'


# In[ ]:


small_sampler_checkpoint = "stabilityai/stable-code-instruct-3b"
small_sampler_tokenizer = AutoTokenizer.from_pretrained(small_sampler_checkpoint)
small_sampler_model = AutoModelForCausalLM.from_pretrained(small_sampler_checkpoint, device_map = 'cuda')


# In[ ]:


print(small_sampler_model)


# In[ ]:


peft_config = LoraConfig(
        lora_alpha=64, # alpha ~= 2r
        lora_dropout=0.05,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)


# In[ ]:


training_arguments = SFTConfig(
        output_dir="./codeGEN_LoRA/",
        eval_strategy="steps",
        do_eval=True,
        optim="paged_adamw_8bit",
        log_level="debug",
        save_strategy="epoch",
        logging_steps=100,
        learning_rate=1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        eval_steps=100,
        num_train_epochs=3,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        dataset_text_field="text",
        max_seq_length=512,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=8,
)


# In[ ]:


class WandbCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        # Log metrics at the end of each epoch
        wandb.log({"epoch": state.epoch, "accuracy": kwargs.get("metrics", {}).get("eval_accuracy", 0)})

trainer = SFTTrainer(
    model=small_sampler_model,
    train_dataset=train,
    eval_dataset=validation,
    peft_config=peft_config,
    tokenizer=small_sampler_tokenizer,
    args=training_arguments,
    callbacks=[WandbCallback()]  # Add the callback here
)


# In[ ]:


trainer.train()


# In[ ]:


model = PeftModel.from_pretrained(small_sampler_model, "./codeGEN_LoRA/checkpoint-150", device_map = 'cuda')


# In[ ]:


def function_calling(model, prompt):
    inputs = small_sampler_tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device.type)
    generation_output = model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(temperature=0.5, top_p=1.0, top_k=50, num_beams=1,eos_token_id = [0] ),
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=100,
    )
    for seq in generation_output.sequences:
        output = small_sampler_tokenizer.decode(seq)
        # print("------")
        # print(output)
        # print("------")
        # print(output.split("<|assistant|>")[1].replace("<|endoftext|>", "").strip())
        # print("------")
        return output.split("<|im_start|>assistant")[1].replace("<|endoftext|>", "").replace("<|im_end|>", "").strip()


# In[ ]:


print(validate[0])
print(train_set[2001]['answers'])


# In[ ]:


function_calling(model, validate[0]['text'])


# In[ ]:


print(test_set[0])


# **Evaluation**

# In[ ]:


use_function = 0
none_needed = 0
already_know = 0
doesnt_know = 0
def safe_parse(data):
    try:
        answer = json.loads(data['answers'])
        return 'name' in answer[0]
    except:
        return false

for data in dataset:
    if data['tools'] != '[]' and safe_parse(data):
        use_function +=1
    elif data['tools'] != '[]' and not safe_parse(data):
        none_needed += 1
    elif data['tools'] == '[]' and not safe_parse(data):
        already_know += 1
    else:
        doesnt_know += 1


print(use_function)
print(none_needed)
print(already_know)
print(doesnt_know)


# In[ ]:


print(dataset[2206])


# In[ ]:


checker = [
    {
      "query": "Can you tell me the current population and GDP of Japan?",
      "id": 1,
      "answers": "[{\"name\": \"get_population\", \"arguments\": {\"country\": \"Japan\"}}, {\"name\": \"get_gdp\", \"arguments\": {\"country\": \"Japan\"}}]",
      "tools": "[{\"name\": \"get_population\", \"description\": \"Retrieve the current population of the specified country.\", \"parameters\": {\"country\": {\"description\": \"The name of the country to retrieve the population for.\", \"type\": \"str\", \"default\": \"USA\"}}}, {\"name\": \"get_gdp\", \"description\": \"Retrieve the current GDP of the specified country.\", \"parameters\": {\"country\": {\"description\": \"The name of the country to retrieve the GDP for.\", \"type\": \"str\", \"default\": \"USA\"}}}]"
    },
    {
        'query': 'What are the benefits of regular exercise?',
        'id': 2,
        'answers': '[]',
        'tools': '[{"name": "get_weather_forecast", "description": "Retrieve the weather forecast from the WeatherAPI based on the specified location.", "parameters": {"location": {"description": "The location to get the weather forecast for (e.g., city name, coordinates).", "type": "str"}}}]'
    },
    {
        'query': 'How many tons of TNT would it take to bring down the SalesForce tower in San Francisco?',
        'id': 3,
        'answers': '[No function calls could be made because the required functions are missing.]',
        'tools': '[]'
    }
    {
        'query': 'How many tons of TNT would it take to bring down the SalesForce tower in San Francisco?',
        'id': 3,
        'answers': '[No function calls could be made because the required functions are missing.]',
        'tools': '[]'
    }
]
docs = [{'text' : get_prompt_use(data['query'],data['tools'])} for data in checker]
category_check = Dataset.from_pandas(DataFrame.from_dict(docs))
print(function_calling(model, category_check['text'][2]))
print(checker[0]['answers'])


# In[ ]:


print(category_check['text'][1])


# In[ ]:


# Define the error classes
ERROR_CLASSES = [
    "Invalid Function Call",
    "Incorrect Parameters",
    "Missing Parameters",
    "Incorrect Output Format",
    "Function Call Order",
    "Function Not Supported",
    "Other"
]

# Function to classify errors based on output and expected result
def classify_error(expected, output):
    if output == "":
        return "Invalid Function Call"
    try:
        output_json = json.loads(output)
        expected_json = json.loads(expected)
    except json.JSONDecodeError:
        return "Incorrect Output Format"

    # Example checks
    if output_json != expected_json:
        if isinstance(output_json, dict) and isinstance(expected_json, dict):
            if output_json.get("name") != expected_json.get("name"):
                return "Function Call Order"
            elif output_json.get("arguments") != expected_json.get("arguments"):
                return "Incorrect Parameters"
        else:
            return "Incorrect Output Format"
    return None




# In[ ]:


def evaluate_model(model, tokenizer, test_data):
    y_true = []
    y_pred = []
    distances = []
    error_details = []

    def record_error_details(prompt, output, expected_output, distance):
        error_details.append(["prompt:", prompt.split("<|im_start|>user\n")[1].replace("<|endoftext|>", "").replace("<|im_end|>", "").strip(),"output:",output,"expected_output:", expected_output,"levenshtein_distance:", distance])

    for example in test_data:
        prompt = get_prompt_use(example['query'], example['tools'])
        output = function_calling(model, prompt)
        expected_output = example['answers']

        error = classify_error(expected_output, output)
        if error:
            y_true.append("Correct")
            y_pred.append(error)
            distance = Levenshtein.distance(expected_output, output)
            distances.append(distance)
            record_error_details(prompt, output, expected_output, distance)
        else:
            y_true.append("Correct")
            y_pred.append("Correct")
            distances.append(0)

    report = classification_report(y_true, y_pred, labels=["Correct"] + ERROR_CLASSES, output_dict=True)
    
    # Initialize Levenshtein distances dictionary
    levenshtein_distances = {error: [] for error in ERROR_CLASSES}
    
    # Add Levenshtein distances to the report
    for i, error in enumerate(y_pred):
        if error != "Correct":
            levenshtein_distances[error].append(distances[i])

    # Compute average Levenshtein distances
    avg_levenshtein_distances = {error: (sum(levenshtein_distances[error]) / len(levenshtein_distances[error])
                                         if levenshtein_distances[error] else 0)
                                 for error in ERROR_CLASSES}

    # Add average Levenshtein distances to the report
    for error in ERROR_CLASSES:
        report[error]['levenshtein_distance'] = avg_levenshtein_distances[error]

    # Convert report to DataFrame for better display
    report_df = pd.DataFrame(report).transpose()

    # Add support column to DataFrame
    report_df['support'] = report_df['support'].fillna(0).astype(int)

    return report_df, error_details


# In[ ]:


example = test_set[0]
prompt = get_prompt_use(example['query'], example['tools'])
print(prompt)
output = function_calling(model, prompt)
expected_output = example['answers']
print(output)
print(expected_output)


# In[ ]:


report, error_details = evaluate_model(model, small_sampler_tokenizer, test_set)


# In[ ]:


print(error_details[0])


# In[ ]:


print(error_details[2])


# In[ ]:


get_ipython().system('pip install numba')


# In[ ]:


from numba import cuda
cuda.select_device(0)
cuda.close()


# In[ ]:


device = cuda.get_current_device()
device.reset()


# In[ ]:


from google.colab import files
files.download("/content")


# In[ ]:


get_ipython().system('tar -cf model.tar.gz /content/codeGEN_LoRA/checkpoint-150')

