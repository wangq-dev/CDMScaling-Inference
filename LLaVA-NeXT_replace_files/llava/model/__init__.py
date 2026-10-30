import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    "llava_gemma": "LlavaGemmaForCausalLM, LlavaGemmaConfig",
    "llava_qwen3": "LlavaQwen3ForCausalLM, LlavaQwen3Config",
    "llava_falcon3": "LlavaFalcon3ForCausalLM, LlavaFalcon3Config",
    "llava_gemma2": "LlavaGemma2ForCausalLM, LlavaGemma2Config",
    "llava_llama3dot2": "LlavaLlama3dot2ForCausalLM, LlavaLlama3dot2Config",
    "llava_qwen3moe": "LlavaQwen3MoeForCausalLM, LlavaQwen3MoeConfig", 
    "llava_deepseekllm": "LlavaDeepSeekLLMForCausalLM, LlavaDeepSeekLLMConfig",
    "llava_deepseekmath":  "LlavaDeepSeekMathForCausalLM, LlavaDeepSeekMathConfig",
    # Add other models as needed
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        print(f"Failed to import {model_name} from llava.language_model.{model_name}. Error: {e}")
