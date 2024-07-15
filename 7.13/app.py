from flask import Flask, request, jsonify, render_template
from openai import OpenAI

app = Flask(__name__)

# 初始化OpenAI客户端
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

@app.route('/')
def index():
    return render_template('UI.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data['question']
    completion = client.chat.completions.create(
        model="YC-Chen/Breeze-7B-Instruct-v1_0-GGUF",
        messages=[
            {"role": "system", "content": "Answer my questions."},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
    )
    answer = completion.choices[0].message.content
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
