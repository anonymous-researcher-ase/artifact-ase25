import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
import os

model_name = "bigcode/starcoder"
train_data_path = "data/java_or_python.jsonl"
output_dir = "checkpoints/starcoder_sft_java_or_python"

def load_data(path):
    import json
    with open(path, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return Dataset.from_list([{ "text": f"### Code:\n{item['code']}\n### Summary:\n{item['summary']}" } for item in data])

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

train_dataset = load_data(train_data_path)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=1024)

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
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