"""Model fine-tuning module using supervised fine-tuning with LoRA.

Provides functionality for fine-tuning vision-language models on custom datasets
using Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters. Integrates with
Modal for serverless GPU compute and Weights & Biases for experiment tracking.
"""

import logging
import os
from pathlib import Path

import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from .artifacts import load_dataset, load_model_and_processor
from .callbacks import ProcessorSaveCallback
from .config import FineTuningConfig
from .data_preparation import format_dataset_as_conversation, split_dataset
from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)
from .paths import get_path_model_checkpoints_in_modal_volume

logger = logging.getLogger(__name__)

modal_app = get_modal_app("car-maker-identification")
docker_img = get_docker_image()
checkpoint_vol = get_volume("models")


def _build_collate_fn(tokenizer_processor):
    """Construct data collator for batch processing during training."""

    def collator(samples):
        batch_data = tokenizer_processor.apply_chat_template(
            samples, tokenize=True, return_dict=True, return_tensors="pt"
        )
        target_labels = batch_data["input_ids"].clone()
        target_labels[target_labels == tokenizer_processor.tokenizer.pad_token_id] = -100
        batch_data["labels"] = target_labels
        return batch_data

    return collator


def _configure_wandb(training_config: FineTuningConfig) -> None:
    """Initialize or disable Weights & Biases tracking."""
    if training_config.use_wandb:
        experiment_id = training_config.wandb_experiment_name or "fine-tune-experiment"
        logger.info("Initializing WandB experiment: %s", experiment_id)
        wandb.init(
            project=training_config.wandb_project_name,
            name=training_config.wandb_experiment_name,
            config=training_config.__dict__,
        )
    else:
        os.environ["WANDB_DISABLED"] = "true"


def _prepare_datasets(training_config: FineTuningConfig) -> tuple[Dataset, Dataset]:
    """Load and prepare training and evaluation datasets."""
    logger.info("Loading dataset: %s", training_config.dataset_name)
    full_dataset = load_dataset(
        dataset_name=training_config.dataset_name,
        splits=training_config.dataset_splits,
        n_samples=training_config.dataset_samples,
        seed=training_config.seed,
    )

    logger.info("Splitting dataset into train/eval with ratio: %.2f", training_config.train_split_ratio)
    train_ds, eval_ds = split_dataset(
        full_dataset,
        test_size=(1 - training_config.train_split_ratio),
        seed=training_config.seed,
    )

    logger.info("Formatting datasets as conversations")
    conversation_params = {
        "system_prompt": training_config.system_prompt,
        "user_prompt": training_config.user_prompt,
        "image_column": training_config.dataset_image_column,
        "label_column": training_config.dataset_label_colum,
        "label_mapping": training_config.label_mapping,
    }
    
    train_formatted = format_dataset_as_conversation(train_ds, **conversation_params)
    eval_formatted = format_dataset_as_conversation(eval_ds, **conversation_params)

    logger.info("Dataset preparation complete")
    logger.info("Training samples: %d", len(train_formatted))
    logger.info("Evaluation samples: %d", len(eval_formatted))
    logger.debug("Train sample: %s", train_formatted[0])
    logger.debug("Eval sample: %s", eval_formatted[0])

    return train_formatted, eval_formatted


def _apply_peft_adapters(base_model, training_config: FineTuningConfig):
    """Apply LoRA PEFT configuration to model."""
    logger.info("Applying LoRA PEFT configuration")
    adapter_config = LoraConfig(
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        r=training_config.lora_r,
        bias="none",
        target_modules=training_config.lora_target_modules,
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(base_model, adapter_config)
    peft_model.print_trainable_parameters()
    return peft_model


def _create_sft_config(training_config: FineTuningConfig, output_path: str) -> SFTConfig:
    """Build supervised fine-tuning configuration."""
    return SFTConfig(
        output_dir=output_path,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=training_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        logging_steps=training_config.logging_steps,
        optim=training_config.optim,
        gradient_checkpointing=True,
        max_length=512,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="wandb" if training_config.use_wandb else None,
        eval_strategy="steps",
        eval_steps=training_config.eval_steps,
        per_device_eval_batch_size=training_config.batch_size,
        save_strategy="steps",
        save_steps=training_config.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )


@modal_app.function(
    image=docker_img,
    gpu="L40S",
    volumes={"/model_checkpoints": checkpoint_vol},
    secrets=get_secrets(),
    timeout=3600,
    retries=get_retries(max_retries=1),
    max_inputs=1,
)
def fine_tune(training_config: FineTuningConfig):
    """Execute supervised fine-tuning of VLM with LoRA adapters.

    Performs end-to-end training workflow including data preparation,
    model loading, PEFT configuration, and training with checkpointing.

    Args:
        training_config: Configuration parameters for training run
    """
    logger.info("Initiating fine-tuning job")

    _configure_wandb(training_config)

    vlm_model, vlm_processor = load_model_and_processor(model_id=training_config.model_name)
    train_data, eval_data = _prepare_datasets(training_config)

    if training_config.use_peft:
        vlm_model = _apply_peft_adapters(vlm_model, training_config)

    batch_collator = _build_collate_fn(vlm_processor)

    checkpoint_dir = get_path_model_checkpoints_in_modal_volume(
        training_config.wandb_experiment_name
    )
    logger.info("Checkpoints directory: %s", checkpoint_dir)

    sft_training_config = _create_sft_config(training_config, checkpoint_dir)
    proc_callback = ProcessorSaveCallback(vlm_processor)

    logger.info("Initializing SFT trainer")
    trainer = SFTTrainer(
        model=vlm_model,
        args=sft_training_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=batch_collator,
        processing_class=vlm_processor.tokenizer,
        callbacks=[proc_callback],
    )

    resume_checkpoint = None
    if training_config.checkpoint_path is not None:
        resume_checkpoint = str(Path("/model_checkpoints") / training_config.checkpoint_path)
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)
    else:
        logger.info("Training from scratch")

    logger.info("Starting supervised fine-tuning")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if training_config.use_wandb:
        wandb.finish()

    logger.info("Fine-tuning completed")


@modal_app.local_entrypoint()
def main(config_file_name: str):
    """Entry point for launching fine-tuning jobs via Modal.

    Loads configuration and executes remote training on serverless GPU.

    Args:
        config_file_name: Path to YAML configuration file
    """
    training_config = FineTuningConfig.from_yaml(config_file_name)

    try:
        fine_tune.remote(config=training_config)
        logger.info("Fine-tuning job completed successfully")
    except Exception as training_error:
        logger.error("Fine-tuning job failed: %s", training_error)
        raise
