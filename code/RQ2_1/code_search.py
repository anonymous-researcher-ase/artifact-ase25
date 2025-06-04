import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 设置变量
model_dir = "checkpoints/starcoder_sft_java_or_python"
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

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)
        hidden_states = output.hidden_states[-1][0]
        return hidden_states.mean(dim=0).cpu().numpy()

def top_k_accuracy(code_embeddings, query_embeddings, k=10):
    sims = cosine_similarity(query_embeddings, code_embeddings)
    topk = np.argsort(-sims, axis=1)[:, :k]
    correct = 0
    for i in range(len(query_embeddings)):
        if i in topk[i]:
            correct += 1
    return correct / len(query_embeddings)

def load_data(path):
    with open(path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return data

for lang, file in low_resource_test_files.items():
    print(f"Evaluating code search on {lang}...")
    data = load_data(file)
    code_vecs = []
    query_vecs = []

    for item in tqdm(data):
        code_emb = get_embedding(item["code"])
        query_emb = get_embedding(item["summary"])
        code_vecs.append(code_emb)
        query_vecs.append(query_emb)

    acc = top_k_accuracy(np.stack(code_vecs), np.stack(query_vecs), k=10)
    print(f"[Top-10 Accuracy] {lang}: {acc:.4f}")