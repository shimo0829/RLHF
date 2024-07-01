import accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import os
import re

torch.random.manual_seed(0)  #將隨機數生成器的種子設為0

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant. Please provide safe, ethical and accurate information to the user"},
    {"role": "user", "content": "Who was the first human on earth?"},
    {"role": "assistant", "content": "Adam"},
    {"role": "user", "content": "解釋量子計算的基本原理。"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 100,
    "return_full_text": False,
    "temperature": 0.5,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
generated_answer = output[0]['generated_text']
print(generated_answer)

cleaned_answer = re.sub(r'\s+', ' ', generated_answer).strip()

question = messages[-1]['content']
new_answer = {
    "text": cleaned_answer
}

json_path = os.path.join('app', 'data', 'questions.json')

os.makedirs(os.path.dirname(json_path), exist_ok=True)

if os.path.exists(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)
else:
    questions = []

question_found = False
for item in questions:
    if item['question'] == question:
        item['answers'].append(new_answer)
        question_found = True
        break

if not question_found:
    questions.append({
        "question": question,
        "answers": [new_answer]
    })

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(questions, f, ensure_ascii=False, indent=4)

print("Saved successfully.")
