from model_service import ModelProvider

# Full set of all available models
ALL_MODELS = [
    # OpenAI models
    {"provider": ModelProvider.OPENAI, "model": "gpt-4o-mini", "name": "GPT-4-Mini"},
    {"provider": ModelProvider.OPENAI, "model": "4o", "name": "GPT-4"},
    {"provider": ModelProvider.OPENAI, "model": "o1", "name": "o1"},
    {"provider": ModelProvider.OPENAI, "model": "o1-mini", "name": "o1-mini"},
    {"provider": ModelProvider.OPENAI, "model": "o3-mini", "name": "o3-mini"},
    # DeepInfra models
    {"provider": ModelProvider.DEEPINFRA, "model": "qwen-reasoning", "name": "Qwen-Reasoning"},
    {"provider": ModelProvider.DEEPINFRA, "model": "qwen-base", "name": "Qwen-Base"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-33-70b", "name": "Llama-3.3-70B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-31-70b-instruct", "name": "Llama-3.1-70B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-31-8b-instruct", "name": "Llama-3.1-8B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "phi-4", "name": "Phi-4"},
    # DeepSeek models
    {"provider": ModelProvider.DEEPSEEK, "model": "r1", "name": "DeepSeek-Reasoner"},
    {"provider": ModelProvider.DEEPSEEK, "model": "v3", "name": "DeepSeek-Chat"}
]

# GPT-only configuration
GPT_MODELS = [
    {"provider": ModelProvider.OPENAI, "model": "gpt-4o-mini", "name": "GPT-4-Mini"},
    {"provider": ModelProvider.OPENAI, "model": "4o", "name": "GPT-4"},
    {"provider": ModelProvider.OPENAI, "model": "o1", "name": "o1"},
    {"provider": ModelProvider.OPENAI, "model": "o1-mini", "name": "o1-mini"},
    {"provider": ModelProvider.OPENAI, "model": "o3-mini", "name": "o3-mini"}
]

# DeepInfra-only configuration
DEEPINFRA_MODELS = [
    {"provider": ModelProvider.DEEPINFRA, "model": "qwen-reasoning", "name": "Qwen-Reasoning"},
    {"provider": ModelProvider.DEEPINFRA, "model": "qwen-base", "name": "Qwen-Base"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-33-70b", "name": "Llama-3.3-70B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-31-70b-instruct", "name": "Llama-3.1-70B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-31-8b-instruct", "name": "Llama-3.1-8B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "phi-4", "name": "Phi-4"}
]

# DeepSeek-only configuration
DEEPSEEK_MODELS = [
    {"provider": ModelProvider.DEEPSEEK, "model": "r1", "name": "DeepSeek-Reasoner"},
    {"provider": ModelProvider.DEEPSEEK, "model": "v3", "name": "DeepSeek-Chat"}
]

# High-performance models configuration
BENCHMARK_MODELS = [
    {"provider": ModelProvider.OPENAI, "model": "gpt-4o-mini", "name": "GPT-4-Mini"},
    {"provider": ModelProvider.OPENAI, "model": "4o", "name": "GPT-4"},
    {"provider": ModelProvider.OPENAI, "model": "o3-mini", "name": "o3-mini"},
    {"provider": ModelProvider.DEEPINFRA, "model": "qwen-reasoning", "name": "Qwen-Reasoning"},
    {"provider": ModelProvider.DEEPINFRA, "model": "qwen-base", "name": "Qwen-Base"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-31-70b-instruct", "name": "Llama-3.1-70B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "llama-31-8b-instruct", "name": "Llama-3.1-8B"},
    {"provider": ModelProvider.DEEPINFRA, "model": "phi-4", "name": "Phi-4"},
    {"provider": ModelProvider.DEEPSEEK, "model": "r1", "name": "DeepSeek-Reasoner"},
    {"provider": ModelProvider.DEEPSEEK, "model": "v3", "name": "DeepSeek-Chat"}
] 