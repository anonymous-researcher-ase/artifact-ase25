
# This script is used to generate code using the Qwen2.5Coder model and evaluate the results.

python code/generate_code_first.py
python code/process_data1.py

langs=(cpp ruby java php javascript go)
for lang in "${langs[@]}"; do
    python code/check_generated_parallel.py --input ../data/${lang}/qwen2.5coder_python_${lang}1_cleaned.json --output ../results/first_trans_evaluate_result/evaluate_result_python_${lang}1.json
    python code/calculate_ca.py --input ../results/first_trans_evaluate_result/evaluate_result_python_${lang}1.json --output ../results/ca_base/ca_result_python_${lang}.json
done

python code/generate_code_second.py
python code/process_data2.py

for lang in "${langs[@]}"; do
    python code/check_generated_parallel.py --input ../data/${lang}/qwen2.5coder_python_${lang}2_cleaned.json --output ../results/first_trans_evaluate_result/evaluate_result_python_${lang}2.json
    python code/calculate_ca.py --input ../results/first_trans_evaluate_result/evaluate_result_python_${lang}2.json --output ../results/ca_inter_trans/ca_result_python_${lang}.json
done
