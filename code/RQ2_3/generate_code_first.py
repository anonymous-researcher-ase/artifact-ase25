from typing import Dict

from polyeval.parsing import parse
from polyeval.eval import ProjectTemplate, EvalStatus, gen_codes, gen_codes_for_single_file, create_project
from polyeval.objects.problem import Problem
from polyeval.generators import create_generator
from tqdm import tqdm
import json
import time
from openai import OpenAI
from openai import APIConnectionError

def gen_prompt(lang: str, problem: Problem):
    generator = create_generator(lang, problem)
    return generator.gen_prompt()

def gen_prompts(lang: str):
    print(f"Loading problem description...")
    with open("/poly-humaneval/benchmark/poly_humaneval.testdsl", "r") as f:  # Adjust the path to your problem description file
        desc_str = f.read()
        problems = parse(desc_str)

    prompts = []
    for i in range(164):
        problem = list(problems.values())[i]
        prompt = gen_prompt(lang, problem)
        prompts.append(prompt)
    return prompts

def translate_code(source_code, source_lang, target_lang, prompt = ""):
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


inter_langs = ["go", "ruby", "php", "javascript", "cpp", "java"]
source_lang = "python"
source_file_path = "./data/poly_humaneval_sol.json"

def translate_first():
    with open(source_file_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    for inter_lang in inter_langs:
        print(f"\n🔁 Translating to: {inter_lang}")
        if inter_lang != "unicode" :
            prompts = gen_prompts(inter_lang)
        else :
            prompts = [[""]]

        translated_data = {source_lang: {inter_lang: []}}

        for idx, (id, source_code) in enumerate(tqdm(data[source_lang].items())):
            if inter_lang != "unicode": 
                translated_code = translate_code(source_code, source_lang, inter_lang, prompts[idx])
            else :
                translated_code = translate_code(source_code, source_lang, inter_lang)
            translated_data[source_lang][inter_lang].append(translated_code)

        output_file_path = f'/poly-humaneval/evaluation/data/{inter_lang}/qwen2.5coder_python_{inter_lang}1.json'
        with open(output_file_path, "w", encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=4)

        print(f"✅ Saved to {output_file_path}")

if __name__ == "__main__":
    translate_first()
