from openai import OpenAI
import os
import json
from pathlib import Path
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

# 初始化 OpenAI 客户端
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 可调参数
max_tasks_per_feature = 100
max_retries = 3
max_workers = 5  # 同时并发语言数

# 当前处理的语言列表（按批处理）
languages = [
    "C++", "Java", "JavaScript", "Kotlin", "Python", "Rust", "Haskell", "C",
    "Go", "Swift", "AppleScript", "Fortran", "Dart", "Ruby", "Raku", "PHP",
    "Visual Basic", "Pascal", "Scala"
]

features = [
    ("F1", "Variable Definition"),
    ("F2", "Conditional Branching"),
    ("F3", "Loop: For"),
    ("F4", "Loop: While"),
    ("F5", "System I/O"),
    ("F6", "Arithmetic Operations"),
    ("F7", "Logical Operations"),
    ("F8", "Comparison Operations"),
    ("F9", "Library Integration"),
    ("F10", "Parameter Passing"),
    ("F11", "Function Returns"),
    ("F12", "Exception Handling"),
    ("F13", "Array Usage"),
    ("F14", "List Usage"),
    ("F15", "Set Usage"),
    ("F16", "Map/Dictionary Usage"),
    ("F17", "Class Definition"),
    ("F18", "Object Creation"),
    ("F19", "Inheritance Mechanism"),
    ("F20", "Functional Map"),
    ("F21", "Functional Filter")
]

task_input_dir = Path("feature_tasks")
output_dir = Path("code_outputs_json_threaded")
output_dir.mkdir(exist_ok=True)

def build_prompt(task, feature_name, lang):
    return f'''Translate the following programming task into {lang} code.
The task emphasizes the feature: "{feature_name}".
Only output the code, no comments or explanation.

Task:
"{task}"
'''

def safe_filename(name):
    return name.replace(" ", "_").replace("/", "_").replace(":", "_")

def process_language(lang):
    lang_dir = output_dir / lang.replace(" ", "_")
    lang_dir.mkdir(exist_ok=True)
    for code, feat_name in features:
        feat_file = task_input_dir / f"{code}_{safe_filename(feat_name)}.json"
        if not feat_file.exists():
            print(f"❌ Missing feature file: {feat_file}")
            continue

        output_path = lang_dir / f"{code}_{safe_filename(feat_name)}.json"
        if output_path.exists():
            print(f"⏭️  Skipping {lang} - {code} (already exists)")
            continue

        with open(feat_file) as f:
            tasks = json.load(f)

        task_code_pairs = []
        for idx, task in enumerate(tasks[:max_tasks_per_feature]):
            prompt = build_prompt(task, feat_name, lang)
            for attempt in range(max_retries):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a multilingual code generation assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        timeout=60
                    )
                    code_only = response.choices[0].message.content.strip()
                    task_code_pairs.append({
                        "task": task,
                        "code": code_only
                    })
                    print(f"✅ [{lang}] {code} Task {idx+1}")
                    break
                except Exception as e:
                    print(f"⚠️  Retry {attempt + 1} failed for [{lang}] {code} Task {idx+1}: {e}")
                    sleep(3)
                    if attempt == max_retries - 1:
                        task_code_pairs.append({
                            "task": task,
                            "code": f"// [ERROR] {str(e)}"
                        })
            sleep(0.5)

        with open(output_path, "w") as f:
            json.dump(task_code_pairs, f, indent=2)
        print(f"💾 Saved to {output_path}")

# 主线程调度器
if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_language, lang) for lang in languages]
        for future in as_completed(futures):
            future.result()  # 触发异常抛出（调试用）
