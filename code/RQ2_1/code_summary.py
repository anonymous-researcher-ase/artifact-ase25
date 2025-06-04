import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from evaluate import load
import json
from tqdm import tqdm

# 设置变量
model_dir = "checkpoints/starcoder_sft_java_or_python"  # Java-SFT 或 Python-SFT
low_resource_test_files = {
    "kotlin": "data/kotlin.jsonl",
    "swift": "data/swift.jsonl",
    "haskell": "data/haskell.jsonl",
    "applescript": "data/applescript.jsonl"
}

# 加载模型与 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
model.eval()

bleu = load("bleu")

def load_data(path):
    with open(path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return Dataset.from_list(data)

def generate_summary(code):
    input_text = f"### Code:\n{code}\n### Summary:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Summary:")[-1].strip()

for lang, file in low_resource_test_files.items():
    dataset = load_data(file)
    predictions = []
    references = []
    for example in tqdm(dataset, desc=f"Evaluating {lang}"):
        pred = generate_summary(example["code"])
        predictions.append(pred)
        references.append([example["summary"]])
    result = bleu.compute(predictions=predictions, references=references)
    print(f"[BLEU] {lang}: {result['bleu']:.4f}")