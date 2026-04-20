"""Configuration Module

This module defines the default configuration parameters for the system, including model settings, chunking settings, and generation parameters.
"""
from __future__ import annotations


# Model Configuration
DEFAULT_MODEL = "qwen3:9b"  # Default LLM model to use
DEFAULT_BASE_URL = "http://localhost:11434"  # Default base URL for the Ollama API

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 220  # Default size of document chunks (in characters)
DEFAULT_CHUNK_OVERLAP = 50  # Default overlap size between document chunks (in characters)
DEFAULT_TOP_K = 4  # Default number of top-k results to return during retrieval
DEFAULT_MEMORY_TURNS = 4  # Default maximum number of turns for conversation history

# Generation Parameters Configuration
DEFAULT_TEMPERATURE = 0.2  # Default temperature parameter, controlling the randomness of generated text
DEFAULT_TOP_P = 0.9  # Default top-p parameter, controlling the diversity of generated text
DEFAULT_NUM_PREDICT = 220  # Default number of tokens to predict

