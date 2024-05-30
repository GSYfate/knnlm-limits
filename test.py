from transformers import AutoTokenizer, AutoModel
# MODEL=meta-llama/Llama-2-7b-hf
# MODEL=meta-llama/Meta-Llama-3-8B
# 加载预训练的模型和分词器
model_name = "mistralai/Mistral-7B-v0.3"  # 你可以替换为你想要使用的模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 输入文本
text = "as the population of plants decreases , carbon in the atmosphere will increase rain"

# 使用分词器进行编码
encoded_input = tokenizer(text, return_tensors='pt')

# 输出编码后的向量
print(encoded_input)
text = "as the population of plants decreases , carbon in the atmosphere will increase"

# 使用分词器进行编码
encoded_input = tokenizer(text, return_tensors='pt')

# 输出编码后的向量
print(encoded_input)
text = " rain"

# 使用分词器进行编码
encoded_input = tokenizer(text, return_tensors='pt')

# 输出编码后的向量
print(encoded_input)
print(model.config._name_or_path)

decoded_text = tokenizer.decode([1, 8064])

# 输出解码后的文本
print("Decoded text:", decoded_text)