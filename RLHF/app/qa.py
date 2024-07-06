import os 
import re
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="YC-Chen/Breeze-7B-Instruct-v1_0-GGUF",
  messages=[
    {"role": "system", "content": "Answer my question."},
    {"role": "user", "content": "解釋量子計算的基本原理。"}
  ],
  temperature=0.7,
)

response = completion.choices[0].message.content

cleaned_answer = re.sub(r'\s+', ' ', response).strip()

question = "解釋量子計算的基本原理。"
new_answer = {
    "text": cleaned_answer
}

json_path = os.path.join('RLHF', 'app', 'data', 'questions.json')

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