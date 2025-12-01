"""
https://modelscope.cn/models/sentence-transformers/all-MiniLM-L6-v2/files
"""
import torch
from sentence_transformers import SentenceTransformer, util

from init import Root

# 1. 加载一个预训练好的模型
model = SentenceTransformer(str(Root / './weight/all-MiniLM-L6-v2'))  # 一个轻量级但效果很好的模型


standard_sentences = ["你好", "早上好", "我喜欢学习英语", ]# 2. 标准句子
input_sentence = "爱学习"# 4. 处理输入句子

# 3. 计算所有标准句子的嵌入
standard_embeddings = model.encode(standard_sentences, convert_to_tensor=True)


input_embedding = model.encode(input_sentence, convert_to_tensor=True)

# 5. 计算余弦相似度
cos_scores = util.cos_sim(input_embedding, standard_embeddings)

# 6. 找到最匹配的
most_similar_idx = torch.argmax(cos_scores).item()
matched_sentence = standard_sentences[most_similar_idx]

print(f"输入: {input_sentence}")
print(f"最匹配: {matched_sentence}")
