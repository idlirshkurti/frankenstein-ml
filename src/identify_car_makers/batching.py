"""Dataset batching utilities for evaluation workflows.

Provides functionality to partition datasets into fixed-size batches of images
and labels for efficient batch processing during model evaluation.
"""

import datasets
from PIL import Image

from .config import EvaluationConfig


def _extract_label(sample: dict, config: EvaluationConfig) -> str:
    """Extract and optionally map label from dataset sample."""
    raw_label = sample[config.label_column]
    
    if config.label_mapping is not None:
        return config.label_mapping[raw_label]
    
    return raw_label


def create_batches(
    dataset: datasets.Dataset, config: EvaluationConfig
) -> list[tuple[list[Image.Image], list[str]]]:
    """Partition dataset into batches for inference.

    Groups dataset samples into fixed-size batches according to configuration.
    Handles label mapping if specified in config. Final batch may be smaller
    than batch_size if dataset size is not evenly divisible.

    Args:
        dataset: HuggingFace dataset to batch
        config: Configuration specifying batch size and column mappings

    Returns:
        List of (images, labels) tuples representing batches
    """
    batch_collection = []
    img_buffer = []
    label_buffer = []

    for data_point in dataset:
        img = data_point[config.image_column]
        lbl = _extract_label(data_point, config)

        img_buffer.append(img)
        label_buffer.append(lbl)

        if len(img_buffer) == config.batch_size:
            batch_collection.append((img_buffer, label_buffer))
            img_buffer = []
            label_buffer = []

    # Include partial batch if remaining samples exist
    if img_buffer:
        batch_collection.append((img_buffer, label_buffer))

    return batch_collection
