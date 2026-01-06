"""Vision-language model evaluation module.

This module provides functionality for running evaluations of VL models
against labeled datasets. Supports both structured and unstructured generation
modes with batch processing capabilities.
"""

import logging
import tempfile
import time

import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from .batching import create_batches
from .config import EvaluationConfig
from .inference import get_model_output, get_structured_model_output
from .artifacts import load_dataset, load_model_and_processor
from .modal_infra import (
    get_docker_image,
    get_modal_app,
    get_retries,
    get_secrets,
    get_volume,
)
from .output_types import CarIdentificationOutputType
from .report import EvalReport

logger = logging.getLogger(__name__)

# Modal infrastructure setup
modal_app = get_modal_app("car-maker-identification")
docker_img = get_docker_image()
dataset_vol = get_volume("datasets")
model_vol = get_volume("models")


def _process_structured_batch(
    imgs_batch, labels_batch, vlm_model, vlm_processor, eval_config, report
):
    """Process a batch using structured generation.

    Returns:
        Number of correct predictions in this batch
    """
    single_sample = len(imgs_batch) == 1

    outputs = _get_batch_outputs(
        imgs_batch, vlm_model, vlm_processor, eval_config, single_sample
    )

    if outputs is None:
        logger.warning("Batch processing failed, skipping %d samples", len(imgs_batch))
        return 0

    return _record_structured_predictions(imgs_batch, labels_batch, outputs, report)


def _get_batch_outputs(
    imgs_batch, vlm_model, vlm_processor, eval_config, single_sample
):
    """Get model outputs for a batch of images."""
    if single_sample:
        structured_output = get_structured_model_output(
            vlm_model,
            vlm_processor,
            eval_config.system_prompt,
            eval_config.user_prompt,
            imgs_batch[0],
        )
        return [structured_output] if structured_output else [None]

    return get_structured_model_output(
        vlm_model,
        vlm_processor,
        eval_config.system_prompt,
        eval_config.user_prompt,
        imgs_batch,
    )


def _record_structured_predictions(imgs_batch, labels_batch, outputs, report):
    """Record structured predictions and calculate correctness.

    Returns:
        Number of correct predictions
    """
    correct = 0
    for img, gt_label, prediction in zip(imgs_batch, labels_batch, outputs):
        if prediction is not None:
            predicted_class = prediction.pred_class
            is_correct = predicted_class == gt_label
            correct += int(is_correct)
            report.add_record(img, gt_label, predicted_class)
    return correct


def _process_unstructured_batch(
    imgs_batch, labels_batch, vlm_model, vlm_processor, eval_config, report
):
    """Process a batch using unstructured generation.

    Returns:
        Number of correct predictions in this batch
    """
    correct = 0
    for img, gt_label in zip(imgs_batch, labels_batch):
        message_history = _build_message_history(eval_config, img)
        predicted_class = get_model_output(vlm_model, vlm_processor, message_history)

        is_correct = predicted_class == gt_label
        correct += int(is_correct)
        report.add_record(img, gt_label, predicted_class)

    return correct


def _build_message_history(eval_config, img):
    """Build conversation message history for model input."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": eval_config.system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": eval_config.user_prompt},
            ],
        },
    ]


def _initialize_wandb(eval_config):
    """Initialize Weights & Biases experiment tracking."""
    run_tags = [
        eval_config.model.split("/")[-1],
        eval_config.dataset.split("/")[-1],
    ]
    wandb.init(
        project=eval_config.wandb_project_name,
        config=eval_config.model_dump(),
        tags=run_tags,
    )


def _log_confusion_matrix(report):
    """Generate and log confusion matrix visualization to wandb."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as matrix_file:
        report.print_confusion_matrix()
        plt.savefig(matrix_file.name, dpi=150, bbox_inches="tight")
        plt.close()
        wandb.log({"confusion_matrix": wandb.Image(matrix_file.name)})


@modal_app.function(
    image=docker_img,
    gpu="L40S",
    volumes={
        "/datasets": dataset_vol,
        "/models": model_vol,
    },
    secrets=get_secrets(),
    timeout=3600,
    retries=get_retries(max_retries=1),
    max_inputs=1,
)
def evaluate(eval_config: EvaluationConfig) -> EvalReport:
    """Execute model evaluation on specified dataset.

    Performs inference on dataset samples and computes metrics including
    accuracy and confusion matrix. Results are logged to Weights & Biases.

    Args:
        eval_config: Configuration parameters for evaluation run

    Returns:
        EvalReport containing predictions and computed metrics
    """
    execution_start = time.time()
    logger.info(
        "Initiating evaluation: model=%s dataset=%s",
        eval_config.model,
        eval_config.dataset,
    )

    _initialize_wandb(eval_config)

    ds = load_dataset(
        dataset_name=eval_config.dataset,
        splits=[eval_config.split],
        n_samples=eval_config.n_samples,
        seed=eval_config.seed,
        cache_dir="/datasets",
    )

    vlm_model, vlm_processor = load_model_and_processor(
        model_id=eval_config.model, cache_dir="/models"
    )

    report = EvalReport()
    batch_list = create_batches(ds, eval_config)

    logger.info(
        "Dataset contains %d samples, split into %d batches (batch_size=%d)",
        len(ds),
        len(batch_list),
        eval_config.batch_size,
    )

    for imgs_batch, labels_batch in tqdm(batch_list, desc="Evaluation progress"):
        if eval_config.structured_generation:
            _process_structured_batch(
                imgs_batch, labels_batch, vlm_model, vlm_processor, eval_config, report
            )
        else:
            _process_unstructured_batch(
                imgs_batch, labels_batch, vlm_model, vlm_processor, eval_config, report
            )

    final_accuracy = report.get_accuracy()
    logger.info("Evaluation accuracy: %.4f", final_accuracy)
    wandb.log({"accuracy": final_accuracy})

    _log_confusion_matrix(report)

    execution_time = time.time() - execution_start
    wandb.log({"total_execution_time_seconds": execution_time})

    logger.info(
        "Evaluation completed in %.2f seconds (%.1f minutes)",
        execution_time,
        execution_time / 60,
    )

    wandb.finish()
    return report


@modal_app.local_entrypoint()
def main(config_file_name: str):
    """Entry point for running evaluations via Modal.

    Loads configuration from YAML file, executes remote evaluation,
    and persists results to disk.

    Args:
        config_file_name: Path to YAML configuration file
    """
    eval_config = EvaluationConfig.from_yaml(config_file_name)
    evaluation_report = evaluate.remote(eval_config)
    results_path = evaluation_report.to_csv()
    logger.info("Evaluation results written to: %s", results_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    eval_cfg = EvaluationConfig()
    logger.info("Loading dataset: %s", eval_cfg.dataset)

    data = load_dataset(
        dataset_name=eval_cfg.dataset,
        splits=[eval_cfg.split],
        n_samples=eval_cfg.n_samples,
        seed=eval_cfg.seed,
        cache_dir=None,
    )

    logger.info("Dataset loaded: %d rows", data.num_rows)

    # Simple evaluation without batching
    num_correct = 0
    for data_sample in data:
        logger.debug("Processing sample with image and label extraction")
        sample_image = data_sample[eval_cfg.image_column]

        try:
            normalized_label = eval_cfg.label_mapping[
                data_sample[eval_cfg.label_column]
            ]
        except KeyError:
            logger.error(
                "Label mapping failed for: %s", data_sample[eval_cfg.label_column]
            )
            breakpoint()

        logger.debug("Sample processing boundary")
