"""Parameter-Efficient Fine-Tuning utilities using LoRA adapters.

Provides functions for configuring and applying LoRA (Low-Rank Adaptation)
to large language models for memory-efficient training.
"""

import logging

from peft import LoraConfig, get_peft_model

logger = logging.getLogger(__name__)


def prepare_peft_model(
    model,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
):
    """Configure model with LoRA adapters for efficient fine-tuning.

    Applies Low-Rank Adaptation to specified model layers, enabling
    parameter-efficient training by only updating a small number of
    trainable parameters.

    Args:
        model: Base model to apply LoRA adapters to
        lora_r: Rank of LoRA decomposition matrices
        lora_alpha: Scaling factor for LoRA updates
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to. If None,
            uses default projection and MLP layers

    Returns:
        PEFT model with LoRA adapters applied
    """
    if target_modules is None:
        target_modules = [
            "q_proj",
            "v_proj",
            "fc1",
            "fc2",
            "linear",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    logger.info("Configuring LoRA PEFT adapters")
    logger.debug("LoRA rank: %d, alpha: %d, dropout: %.3f", lora_r, lora_alpha, lora_dropout)
    logger.debug("Target modules: %s", target_modules)

    adapter_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, adapter_config)
    peft_model.print_trainable_parameters()
    
    logger.info("LoRA adapters applied successfully")
    
    return peft_model
