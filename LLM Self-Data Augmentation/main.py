import requests
import json
import os
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import random
import time
import openai_api

prompt_z = "请抽取出文本中的事件，如果有的元素抽取不出来请置为\"\"，" \
           '输出严格遵守如下的json具体格式输出：[{"class": "事件类别","actor": "行为方","action": "行为","object": "受体","time": "时间","location": "地点"},…]\n' \
           "文本如下：\n"

def ask_openai(question):
    response= openai_api.query4(question)
    return response

def ask(question,history=[],url="http://127.0.0.1:8012/ask"):
    data = {'question': question, 'history': history}
    response = requests.post(url, json=data)
    return response.json()['answer']

def load_data():
    with open("./data/p0_gt.json",'r',encoding='utf-8') as f:
        data=json.load(f)

    ret=[]
    for item in data:
        ret.append((item["text"],item['event_list']))

    
    random.seed(42)
    random.shuffle(ret)
    
    return ret

def load_data2():
    with open("./data/5.json",'r',encoding='utf-8') as f:
        data=json.load(f)

    ret=[]
    for item in data:
        ret.append((item["text"],item['casuality_list']))

    to_pop=[]
    for item in ret:
        if len(item[1])>=4:
            to_pop.append(item)
    
    for item in to_pop:
        ret.remove(item)
        
    random.seed(42)
    random.shuffle(ret)
    
    return ret

def list2jsonl(list_):
    return_str=""
    system="你是一个因果分析专家，擅长发现文本中的因果关系。"
    for i,item in enumerate(list_):
        print(i)
        q=item[0]
        a=item[1]
        tmp_dict={"type": "chatml","messages": [{"role": "system","content": system},{"role": "user","content": q},{"role": "assistant","content": a}],"source": "unknown"}
        return_str += json.dumps(tmp_dict,ensure_ascii=False) + "\n"
    return return_str

def test_hello_route():
    # 设置目标 URL
    url = 'http://127.0.0.1:8012/hello'
    
    # 发送 GET 请求
    try:
        response = requests.get(url)
    except:
        return False
    
    # 检查响应状态码
    if response.status_code == 200:
        return True
    else:
        return False

def start_model():
    # 设置目标 URL
    url = 'http://127.0.0.1:8016/start'
    
    # 发送 GET 请求
    response = requests.get(url)
    
    # 检查响应状态码
    if response.status_code == 200:
        print("请求成功！")
        # 打印响应内容（JSON）
        print("响应内容：", response.json())
    else:
        print("请求失败，状态码：", response.status_code)

to_eval=['/home/ubri/llm/qwen/sft/output_qwen/q72_5000_1e4_3/checkpoint-84','/home/ubri/llm/qwen/sft/output_qwen/q72_5000_1e4_3/checkpoint-168',
         '/home/ubri/llm/qwen/sft/output_qwen/q72_5000_1e4_3/checkpoint-252','/home/ubri/llm/qwen/sft/output_qwen/q72_5000_1e4_3/checkpoint-336',
         '/home/ubri/llm/qwen/sft/output_qwen/q72_5000_1e4_3/checkpoint-420','/home/ubri/llm/qwen/sft/output_qwen/q72_5000_1e4_3/checkpoint-504',
         '/home/ubri/llm/qwen/sft/output_qwen/q72_5000_1e4_3/checkpoint-588','/home/ubri/llm/qwen/sft/output_qwen/q72_5000_1e4_3/checkpoint-672',
         '/home/ubri/llm/qwen/sft/output_qwen/q72_5000_1e4_3/checkpoint-756','/home/ubri/llm/qwen/sft/output_qwen/q72_5000_1e4_3/checkpoint-840']

to_save=['/home/ubri/llm/tmp1/output/20/','/home/ubri/llm/tmp1/output/21/',
         '/home/ubri/llm/tmp1/output/22/','/home/ubri/llm/tmp1/output/23/',
         '/home/ubri/llm/tmp1/output/24/','/home/ubri/llm/tmp1/output/25/',
         '/home/ubri/llm/tmp1/output/26/','/home/ubri/llm/tmp1/output/27/',
         '/home/ubri/llm/tmp1/output/28/','/home/ubri/llm/tmp1/output/29/',]

def main4():
    data=load_data()[:100]
    for i in range(len(to_eval)):
        with open('/home/ubri/llm/qwen/change_model/file_path.txt','w',encoding='utf-8') as f:
            f.write(to_eval[i])
        start_model()
        while True:
            if test_hello_route():
                break
            time.sleep(1)
        if not os.path.exists(to_save[i]):
            os.mkdir(to_save[i])
        for j,item in enumerate(data):
            print(i,j)
            # print(item[0])
            answer=ask(prompt_z+item[0])
            with open(to_save[i]+str(j)+'.txt','w',encoding='utf-8') as f:
                f.write(answer)
            print(answer)
            

def main():#问问题
    data=load_data()[:100]
    to_do=["./sft/output_qwen/q72_5000_1e4_2/checkpoint-84"]
    to_save=["./output/20/"]
    for ii in range(len(to_do)):
        if not os.path.exists(to_save[ii]):
            os.mkdir(to_save[ii])
        for i,item in enumerate(data):
            print(i)
            print(item[0])


def main2():  # 生成数据集

    prompt_p1 = """%EVENT_EXTRACTION%
上述内容是JSON格式的从文本中提取的几个事件。请根据以下要求进行重写提取：
1. 保持原有的JSON格式。JSON对象的字段(class, actor, action, object, time, location)不得更改，但JSON数组的大小可以修改。
2. 请修改事件的 "class"（事件类别），并相应地修改 "actor"（行为方）, "action"（行为）, "object"（受体）, "time"（时间）, 和 "location"（地点）。
3. 修改后的 "actor" 和 "object" 必须在整个JSON中保持一致（如果适用）。
4. 修改后的 "time" 和 "location" 必须是合理的。
5. 修改后的JSON必须描绘一个完整的、逻辑连贯的场景，其中每个事件都是这个场景的一部分。
6. 请只输出JSON格式，确保不包含任何多余的内容。

请开始重写：
"""

    # 提示2 (基于论文 P2'): 从修改后的JSON重建文本
    prompt_p2 = """%EVENT_EXTRACTION%
上述内容是JSON格式的从文本中提取的几个事件。请根据以下要求重建原始文本片段：
1. 重建的文本必须包含JSON中的所有信息，不得遗漏。
2. JSON中的每一条信息都必须完整地出现在文本片段中，不得被截断或打断。
3. 重建的文本片段长度必须超过200个字符。您可以包含一些不描述事件的句子来满足字数要求。
4. 重建的文本片段必须语义连贯、完整，确保整个文本片段处于一个一致的场景中。
5. 请只输出重建的文本片段，不包含任何多余的内容。

请开始重建文本：
"""

    print("Loading data...")
    data = load_data()
    if not data:
        print("No data loaded. Exiting main2.")
        return

    # 确保数据量足够
    if len(data) < 100:
        print(f"Error: Not enough data. Loaded {len(data)}, but need at least 100.")
        return

    # 1. 分离训练集和测试集
    # 保持与原始代码一致的分割
    test_data = data[:100]
    train_data = data[100:5140]  # 假设原始数据至少有5140条

    print(f"Loaded {len(data)} total samples.")
    print(f"Test set size: {len(test_data)}")
    print(f"Train set size: {len(train_data)}")

    # 2. 处理和保存测试集 (与原始逻辑相同)
    list_test = []
    list_test_gt = []

    print("Processing test set...")
    for i, item in enumerate(test_data):
        q = prompt_z + item[0]
        a = json.dumps(item[1], ensure_ascii=False)
        list_test.append((q, a))
        list_test_gt.append(item[1])

    output_list_test = list2jsonl(list_test)
    with open(f'./data/test_2.jsonl', 'w', encoding='utf-8') as f:
        f.write(output_list_test)

    with open(f'./data/gt_2.jsonl', 'w', encoding='utf-8') as f:
        for item in list_test_gt:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("Test set saved.")

    # 3. 处理和增强训练集
    list_train_augmented = []

    # 3a. 添加所有原始训练数据
    print("Adding original training data...")
    for i, item in enumerate(train_data):
        q = prompt_z + item[0]
        a = json.dumps(item[1], ensure_ascii=False)
        list_train_augmented.append((q, a))

    print(f"Added {len(train_data)} original samples to training set.")

    # 3b. 添加增强数据
    print(f"Starting self-data augmentation for {len(train_data)} samples...")
    print("This will take a long time as it requires 2 LLM calls per sample.")

    for i, item in enumerate(train_data):
        print(f"Augmenting sample {i + 1}/{len(train_data)}...")
        try:
            # 准备阶段1：修改事件JSON
            original_event_json_str = json.dumps(item[1], ensure_ascii=False)
            prompt1_full = prompt_p1.replace("%EVENT_EXTRACTION%", original_event_json_str)

            # 调用LLM（阶段1）
            modified_event_json_str = ask_openai(prompt1_full)

            if not modified_event_json_str:
                print(f"Warning: LLM call 1 (Modify) returned None for sample {i}. Skipping.")
                continue

            # 简单的JSON验证
            try:
                # 确保它至少是有效的JSON
                json.loads(modified_event_json_str)
                # 确保它是一个列表
                if not modified_event_json_str.strip().startswith("["):
                    print(f"Warning: LLM call 1 (Modify) did not return a JSON list for sample {i}. Skipping.")
                    print(f"Received: {modified_event_json_str}")
                    continue
            except json.JSONDecodeError as e:
                print(f"Warning: LLM call 1 (Modify) returned invalid JSON for sample {i}: {e}. Skipping.")
                print(f"Received: {modified_event_json_str}")
                continue

            # 准备阶段2：重建文本
            prompt2_full = prompt_p2.replace("%EVENT_EXTRACTION%", modified_event_json_str)

            # 调用LLM（阶段2）
            new_text = ask_openai(prompt2_full)

            if not new_text or new_text.strip().startswith("["):
                print(f"Warning: LLM call 2 (Reconstruct) returned None or invalid text for sample {i}. Skipping.")
                print(f"Received: {new_text}")
                continue

            # 格式化并添加新的增强样本
            q_aug = prompt_z + new_text
            a_aug = modified_event_json_str  # 答案已经是JSON字符串了

            list_train_augmented.append((q_aug, a_aug))

            if (i + 1) % 100 == 0:
                print(
                    f"Checkpoint: Completed {i + 1} augmentations. Total training samples: {len(list_train_augmented)}")

        except Exception as e:
            print(f"Error processing augmentation for sample {i}: {e}. Skipping.")
            # 建议在长时间运行时添加更稳健的重试逻辑
            time.sleep(1)  # 发生错误时稍作等待

    # 4. 保存增强后的训练集
    print(f"Augmentation complete. Total augmented training samples: {len(list_train_augmented)}")
    print("Saving augmented training set...")
    output_list_train = list2jsonl(list_train_augmented)

    # 保存到新文件以避免覆盖原始文件
    with open(f'./data/train5000_augmented_2.jsonl', 'w', encoding='utf-8') as f:
        f.write(output_list_train)

    print("Augmented training set saved to ./data/train5000_augmented_2.jsonl")
    print(f"Original total data length: {len(data)}")
    print(f"Final augmented training set size (lines): {len(list_train_augmented)}")

def worker(item, index, model_dir, save_dir):
    
    print(index)
    answer = ask(prompt_z + item[0])  # 生成答案
    file_path = os.path.join(save_dir, f'{index}.txt')
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(answer)


def main3():  # 问问题
    data = load_data()
    to_do = ["./sft/output_qwen/q72_5000_1e4_2/checkpoint-84"]
    to_save = ["./output/p0_gpt4/"]
    for ii in range(len(to_do)):
        if not os.path.exists(to_save[ii]):
            os.mkdir(to_save[ii])
        for i, item in enumerate(data):
            print(i)
            answer = ask_openai(prompt_z + item[0])
            if answer is not None:
                with open(to_save[ii] + str(i) + '.txt', 'w', encoding='utf-8') as f:
                    json.dump(answer, f, ensure_ascii=False, indent=4)
            print(answer)

if __name__=='__main__':
    main4()
