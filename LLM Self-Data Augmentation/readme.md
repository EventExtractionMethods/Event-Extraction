# **基于大型语言模型自数据增强的事件抽取**

## **1\. 简介**

本论文核心思想是利用大型语言模型（LLM）的自数据增强能力来扩充训练数据，然后使用增强后的数据对模型（如 Qwen）进行监督式微调（SFT）。项目不仅包含了数据增强和模型微调的代码，还实现了一个轻量级的 Flask API 来部署微调后的 LoRA 模型，以及一个评估脚本来计算抽取结果的精确率、召回率和 F1 分数。

## **2\. 文件结构**

.  
├── main.py             \# 实验主入口：用于数据增强、调用模型生成结果  
├── score.py            \# 评估脚本：计算 P/R/F1 分数  
├── data/                 \# 存放原始数据和增强数据  
│   └── ...  
├── output/             \# 存放模型生成的预测结果  
│   └── ...  
├── finetune/  
│   ├── apilora.py        \# LoRA 模型的 API 接口 (Flask)  
│   ├── init.py           \# 模型初始化脚本  
│   ├── start\_lora.py     \# 启动 API 服务的管理脚本  
│   └── sft/  
│       ├── finetune.py   \# 监督式微调 (SFT) 脚本  
│       └── test.py       \# 简单的测试脚本  
└── README.md           \# 本文档

## **3\. 主要模块功能**

* **main.py**  
  * **数据增强 (main2)**：实现了论文中的“反向数据增强” (Reverse Data Augmentation) 流程。它调用 ask\_openai 接口（需配置 openai\_api.py），先修改原始事件 JSON (Prompt P1')，再让 LLM 根据修改后的 JSON 重建文本 (Prompt P2')，从而生成新的 (text, event\_list) 数据对。  
  * **批量推理 (main3, main4)**：加载测试数据，调用已部署的模型 API (apilora.py) 或 OpenAI API 进行提问，并将结果保存到 output/ 目录以便后续评估。  
* **finetune/sft/finetune.py**  
  * 核心的模型微调脚本。  
  * 基于 transformers 和 peft 库，使用 LoRA (Low-Rank Adaptation) 技术对大模型（如 Qwen）进行监督式微调。  
  * 它会加载 main.py 生成的 jsonl 格式数据，并执行训练过程，最终保存模型的 checkpoint。  
* **finetune/apilora.py**  
  * 一个 Flask Web 服务器，用于加载微调后的 LoRA 模型 checkpoint。  
  * 提供 /ask 接口，接收文本输入，并返回模型的推理结果。  
* **finetune/start\_lora.py**  
  * 用于管理和启动 apilora.py 服务的脚本。  
  * 它提供 /start 接口，可以在不中断主服务的情况下，重新加载或切换 apilora.py 所使用的模型 checkpoint（通过修改 file\_path.txt 实现）。  
* **score.py**  
  * 评估脚本，用于对比模型的预测结果和“黄金标准” (Ground Truth) 答案。  
  * 它会加载 data/ 中的 gt\_...jsonl 和 output/ 中的预测文件。  
  * 实现了 best\_match 贪心算法和字符串相似度（LCS、Levenshtein）来计算事件抽取任务的**精确率 (Precision)**、**召回率 (Recall)** 和 **F1 分数**。

## **4\. 基本使用流程**

### **步骤 1: 数据增强**

1. 准备好原始数据集。  
2. 配置 openai\_api.py（或 main.py 中的 ask\_openai 函数）使其能够访问一个强大的 LLM（如 GPT-4）。  
3. 运行 main.py 中的 main2 函数：  
   \# (可能需要修改 main.py 的 \_\_name\_\_ \== '\_\_main\_\_' 部分来调用 main2)  
   python main.py 

4. 此步骤将生成增强后的训练数据（如 data/train5000\_augmented\_2.jsonl）和测试数据。

### **步骤 2: 模型微调**

1. 使用 finetune/sft/finetune.py 脚本来微调模型。  
   \# 示例命令（具体参数请参照 finetune.py 中的 TrainingArguments）  
   deepspeed finetune/sft/finetune.py \\  
       \--model\_name\_or\_path Qwen/Qwen-7B-Chat \\  
       \--data\_path ./data/train5000\_augmented\_2.jsonl \\  
       \--output\_dir ./finetune/sft/output\_qwen/my\_model \\  
       \--num\_train\_epochs 5 \\  
       \--per\_device\_train\_batch\_size 8 \\  
       \--use\_lora True \\  
       \--q\_lora True \\  
       \--gradient\_checkpointing True \\  
       \--deepspeed ./deepspeed\_config.json 

2. 训练完成后，LoRA checkpoint 将保存在 output\_dir 中。

### **步骤 3: 启动模型 API**

1. 修改 finetune/apilora.py，将 from\_pretrained 的路径指向你训练好的 checkpoint (如 ./sft/output\_qwen/my\_model/checkpoint-xxx)。  
2. 首先启动模型管理服务：  
   python finetune/start\_lora.py

3. 然后访问 http://127.0.0.1:8016/start 来启动 apilora.py 服务（该服务运行在 8012 端口）。

### **步骤 4: 推理与评估**

1. 运行 main.py 中的 main4 函数（确保 ask 函数的 URL 为 http://127.0.0.1:8012/ask）。  
   \# (可能需要修改 main.py 的 \_\_name\_\_ \== '\_\_main\_\_' 部分来调用 main4)  
   python main.py

2. 该脚本会调用在 8012 端口运行的模型 API，对测试集进行推理，并将结果保存到 output/ 目录。  
3. 最后，运行 score.py 来计算最终得分。  
   \# (修改 score.py 中 evaluate 函数的文件路径)  
   python score.py

   输出示例：  
   Precision=XX.XX%, Recall=XX.XX%, F1=XX.XX%