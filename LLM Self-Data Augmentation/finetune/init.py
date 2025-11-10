from transformers import AutoConfig, AutoModelForCausalLM

# 定义模型配置
config = AutoConfig.from_pretrained("./qwen0.5b_15")  # 示例：假设Qwen1.5_0.5B的配置类似于GPT-2，但具体层数和尺寸需要你根据实际情况设置

# 随机初始化模型
model = AutoModelForCausalLM.from_config(config)

# 保存未训练的模型
model.save_pretrained('./tmp')

# 保存配置（如果需要）
config.save_pretrained('./tmp')
