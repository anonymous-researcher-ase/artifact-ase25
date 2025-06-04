import os
import shutil

from polyeval.parsing import parse
from polyeval.eval import ProjectTemplate, EvalStatus, gen_codes, gen_codes_for_single_file, create_project
from polyeval.objects.problem import Problem
from polyeval.generators import create_generator
from tqdm import tqdm
import json
import time
from openai import OpenAI
from openai import APIConnectionError
from concurrent.futures import ThreadPoolExecutor
import os
import shutil


def gen_prompt(lang: str, problem: Problem):
    generator = create_generator(lang, problem)
    return generator.gen_prompt()

def gen_prompts(lang: str):
    print(f"Loading problem description...")
    with open("/poly-humaneval/benchmark/poly_humaneval.testdsl", "r") as f:
        desc_str = f.read()
        problems = parse(desc_str)

    prompts = []
    for i in range(164):
        problem = list(problems.values())[i]
        prompt = gen_prompt(lang, problem)
        prompts.append(prompt)
    return prompts

def translate_code(source_code, source_lang, target_lang, prompt):
    content = \
f"""```{source_code}```\nTranslate the above {source_lang} code into {target_lang} code.
Only output the function definition, including the function signature and body.
Do not include any import statements, comments, main functions, print statements, or any other code.
The function signature is as follows:
{prompt}"""

    client = OpenAI(
        base_url="http://localhost:11434/v1",  # adjust the base URL if needed
        api_key="test"
    )

    while True:
        try:
            completion = client.chat.completions.create(
                model="qwen2.5-coder:14b",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": content}
                ]
            )
            return completion.choices[0].message.content
        except APIConnectionError as e:
            print(f"\nAPI Connection Error: {e}")
            print("Waiting 60 seconds before retrying...")
            time.sleep(60)
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            raise


def translate_second(output_file_path1, evaluate_result_path, source_lang, inter_lang, target_langs, output_file_path2):

    with open(source_file_path, "r", encoding='utf-8') as f:
        source_data = json.load(f)

    with open(output_file_path1, "r", encoding='utf-8') as f:
        data = json.load(f)

    with open(evaluate_result_path) as f:
        evaluate_result = json.load(f)

    translated_data = {source_lang: {}}

    for target_lang in target_langs:
        print(f"\n=== Translating to {target_lang} ===")
        translated_data[source_lang][target_lang] = []

        prompts = gen_prompts(target_lang)

        for idx, inter_code in enumerate(tqdm(data[source_lang][inter_lang])):
            if evaluate_result[source_lang][inter_lang][idx]:
                translated_code = translate_code(inter_code, inter_lang, target_lang, prompts[idx])
            else:
                source_code = list(source_data[source_lang].values())[idx]
                translated_code = translate_code(source_code, source_lang, target_lang, prompts[idx])
            translated_data[source_lang][target_lang].append(translated_code)

    with open(output_file_path2, "w", encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ All translations saved to {output_file_path2}")


if __name__ == "__main__":
    inter_langs = ["go", "ruby", "php", "javascript", "cpp", "java"]
    source_lang = "python"
    source_file_path = "./data/poly_humaneval_sol.json"
    target_langs = [
        {"go" : ["ruby", "php", "javascript", "cpp", "java"]},
        {"ruby" : ["go", "php",  "javascript", "cpp", "java"]}, 
        {"php" : ["go", "ruby",  "javascript", "cpp", "java"]},
        {"javascript" : ["go", "ruby", "php", "cpp","java"]},
        {"cpp" : ["go", "ruby", "php", "javascript", "java"]},
        {"java" : ["go", "ruby", "php", "javascript", "cpp"]}
        ]

    for inter_lang in inter_langs:
        output_file_path1 = f'../data/{inter_lang}/qwen2.5coder_python_{inter_lang}1_cleaned.json'   # Adjusted path to cleaned data
        evaluate_result_path = f"../results/first_trans_evaluate_result/evaluate_result_python_{inter_lang}1.json" # Adjusted path to evaluation results
        output_file_path2 = f"../data/{inter_lang}/qwen2.5coder_python_{inter_lang}2.json" # Adjusted path for second translation output
        translate_second(output_file_path1, evaluate_result_path, source_lang, inter_lang, list(target_langs[inter_langs.index(inter_lang)].values())[0], output_file_path2)
