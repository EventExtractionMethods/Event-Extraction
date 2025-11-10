from flask import Flask, request, jsonify
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer 
import gc

app = Flask(__name__)
# 加载第一个模型和分词器
model = AutoPeftModelForCausalLM.from_pretrained("./sft/output_qwen/1/checkpoint-600", device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("./sft/output_qwen/1/checkpoint-600")
device="cuda"

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'msg': 'ok'}), 200


@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    system = data.get('system','You are a helpful assistant.')
    history = data.get('history')
    print("Q:" + question)
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        messages = [
            {"role": "system", "content": system}
        ]
        for item in history:
            messages.append({'role': 'user', 'content':item[0]})
            messages.append({'role': 'bot', 'content':item[1]})
        messages.append({'role': 'user', 'content':question})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=4096
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("A:" + response)
        return jsonify({'answer': response})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
        
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8012)
