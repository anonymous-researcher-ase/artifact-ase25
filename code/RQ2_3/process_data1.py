import os
import re
import json

def process_file(langs):
    for lang in langs:
        input_file_path = f'../data/{lang}/qwen2.5coder_python_{lang}1.json'
        output_file_path = f'../data/{lang}/qwen2.5coder_python_{lang}1_cleaned.json' 

        def extract_code_blocks(text, target_lang):
            pattern = rf"```{target_lang}\n(.*?)```"
            match = re.search(pattern, text, re.DOTALL)
            return match.group(1) if match else text

        with open(input_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        extracted_data = {}
        for inter_lang, translations in data.items():
            extracted_data[inter_lang] = {}
            for target_lang, code_list in translations.items():
                extracted_data[inter_lang][target_lang] = []
                for code in code_list:
                    code_block = extract_code_blocks(code, target_lang)
                    extracted_data[inter_lang][target_lang].append(code_block)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)

        print(f"Extracted code blocks saved to {output_file_path}")

langs = ["go", "ruby", "php", "javascript", "cpp", "java"]
process_file(langs)