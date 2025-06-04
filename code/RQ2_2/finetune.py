import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import os
import json

model_name = "meta-llama/Llama-2-7b-hf"
data_dir = "data/benchmark_curriculum/"
output_dir = "checkpoints/llama2_codegen_sft"

def load_curriculum_languages(language_list):
    all_data = []
    for lang in language_list:
        file_path = os.path.join(data_dir, f"{lang}.jsonl")
        with open(file_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                all_data.append({"text": item["code"]})
    return Dataset.from_list(all_data)

curriculum_languages = ["applescript", "python", "swift", "kotlin", "javascript", "go", "rust", "java", "c++", "haskell"]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

train_dataset = load_curriculum_languages(curriculum_languages)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=1024)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    save_total_limit=2,
    save_strategy="epoch",
    logging_dir=f"{output_dir}/logs",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)