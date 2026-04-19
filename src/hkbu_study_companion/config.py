"""配置模块

该模块定义了系统的默认配置参数，包括模型设置、分块设置和生成参数等。
"""
from __future__ import annotations


# 模型配置
DEFAULT_MODEL = "qwen3:9b"  # 默认使用的LLM模型
DEFAULT_BASE_URL = "http://localhost:11434"  # Ollama API的默认基础URL

# 分块配置
DEFAULT_CHUNK_SIZE = 220  # 文档分块的默认大小（字符数）
DEFAULT_CHUNK_OVERLAP = 50  # 文档分块之间的默认重叠大小（字符数）
DEFAULT_TOP_K = 4  # 检索时返回的默认top-k结果数
DEFAULT_MEMORY_TURNS = 4  # 对话历史的默认最大轮次数

# 生成参数配置
DEFAULT_TEMPERATURE = 0.2  # 默认温度参数，控制生成文本的随机性
DEFAULT_TOP_P = 0.9  # 默认顶部p参数，控制生成文本的多样性
DEFAULT_NUM_PREDICT = 220  # 默认预测的token数量
