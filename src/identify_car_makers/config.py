"""Configuration schemas for model training and evaluation workflows."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings

from .paths import get_path_to_configs

logger = logging.getLogger(__name__)


class FineTuningConfig(BaseSettings):
    """Settings for fine-tuning vision-language models with LoRA."""

    seed: int = 23
    use_wandb: bool = True
    modal_app_name: str

    model_name: str = "LiquidAI/LFM2-VL-450M"
    max_seq_length: int = 2048
    checkpoint_path: str | None = None

    dataset_name: str
    dataset_samples: int
    dataset_image_column: str
    dataset_label_colum: str
    dataset_splits: list[str] = ["train"]
    label_mapping: dict[Any, str] | None = None
    train_split_ratio: float
    preprocessing_workers: int = 2

    system_prompt: str
    user_prompt: str

    use_peft: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"
    use_rslora: bool = False
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    learning_rate: float
    num_train_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    optim: str = "adamw_8bit"
    warmup_ratio: float
    weight_decay: float
    logging_steps: int
    eval_steps: int

    wandb_project_name: str = "car-maker-identification-fine-tuning"
    wandb_experiment_name: str | None = None
    skip_eval: bool = False
    output_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, file_name: str) -> Self:
        """Load configuration from YAML file in configs directory."""
        config_path = Path(get_path_to_configs()) / file_name
        logger.info("Reading configuration from: %s", config_path)

        with open(config_path) as config_file:
            config_data = yaml.safe_load(config_file)

        return cls(**config_data)

    @model_validator(mode="after")
    def set_experiment_name(self):
        """Auto-generate experiment name if not provided."""
        if self.wandb_experiment_name is not None:
            return self

        time_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_identifier = self.model_name.split("/")[-1]
        self.wandb_experiment_name = (
            f"{model_identifier}-{self.dataset_name}-{time_suffix}"
        )

        return self


class EvaluationConfig(BaseSettings):
    """Configuration parameters for model evaluation runs."""

    seed: int = 23
    batch_size: int = 1
    wandb_project_name: str = "car-maker-identification-evals"

    model: str
    structured_generation: bool

    dataset: str
    split: str
    n_samples: int
    image_column: str
    label_column: str
    label_mapping: dict | None = None

    system_prompt: str
    user_prompt: str

    @classmethod
    def from_yaml(cls, file_name: str) -> "EvaluationConfig":
        """Load evaluation configuration from YAML file."""
        yaml_path = Path(get_path_to_configs()) / file_name
        logger.info("Loading evaluation config from: %s", yaml_path)

        with open(yaml_path) as yaml_file:
            params = yaml.safe_load(yaml_file)

        return cls(**params)
