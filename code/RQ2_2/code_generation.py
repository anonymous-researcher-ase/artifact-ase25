import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from evaluate import load
import json
from tqdm import tqdm

model_dir = "checkpoints/llama2_codegen_sft"
test_data_dir = "data/benchmark_curriculum/"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
model.eval()

codebleu = load("codebleu")

test_languages = ["applescript", "python", "swift", "kotlin", "javascript", "go", "rust", "java", "c++", "haskell"]

def load_data(file_path):
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return data

def generate_code(prompt):
    input_text = f"{prompt}\n# Solution:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("# Solution:")[-1].strip()

tokenizer.pad_token = tokenizer.eos_token

for lang in test_languages:
    file_path = f"{test_data_dir}/{lang}.jsonl"
    dataset = load_data(file_path)
    predictions = []
    references = []
    for item in tqdm(dataset, desc=f"Evaluating {lang}"):
        code = generate_code(item["prompt"])
        predictions.append(code)
        references.append([item["reference"]])
    result = codebleu.compute(predictions=predictions, references=references)
    print(f"[CodeBLEU] {lang}: {result['codebleu']:.4f}")