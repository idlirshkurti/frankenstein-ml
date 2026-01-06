"""Dataset and model loading utilities with persistent caching.

Provides functions for loading HuggingFace datasets and VLM models with
intelligent caching to Modal volumes. Handles authentication, configuration
fixes, and persistent storage for efficient reuse across runs.
"""

import json
import logging
import os
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset as hf_load_dataset
from huggingface_hub import login
from transformers import AutoModelForImageTextToText, AutoProcessor

logger = logging.getLogger(__name__)


def _apply_config_fix(model_path: str) -> None:
    """Patch model config.json to fix incompatible model_type identifier."""
    config_file = Path(model_path) / "config.json"

    with open(config_file) as f:
        config_data = json.load(f)

    if config_data.get("model_type") == "lfm2-vl":
        logger.info("Applying config fix for model at: %s", model_path)
        config_data["model_type"] = "lfm2_vl"

        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.debug("Config patched successfully")


def _load_split_from_cache(split_path: Path, dataset_id: str, split_name: str) -> Dataset | None:
    """Attempt to load dataset split from cached location."""
    if not split_path.exists():
        return None

    logger.info("Loading cached dataset: %s split=%s", dataset_id, split_name)
    try:
        ds = Dataset.load_from_disk(str(split_path))
        logger.info("Loaded %d samples from cache", len(ds))
        return ds
    except Exception as load_error:
        logger.warning("Cache load failed: %s", load_error)
        return None


def _download_split(dataset_id: str, split_name: str) -> Dataset:
    """Download dataset split from HuggingFace hub."""
    logger.info("Downloading dataset: %s split=%s", dataset_id, split_name)
    return hf_load_dataset(dataset_id, split=split_name, num_proc=1)


def _cache_dataset_split(ds: Dataset, cache_path: Path) -> None:
    """Persist dataset split to disk cache."""
    logger.info("Caching dataset to: %s", cache_path)
    try:
        ds.save_to_disk(str(cache_path))
        logger.debug("Dataset cached successfully")
    except Exception as cache_error:
        logger.warning("Failed to cache dataset: %s", cache_error)


def load_dataset(
    dataset_name: str,
    splits: list[str],
    n_samples: int | None = None,
    seed: int | None = 42,
    cache_dir: str = "/datasets",
) -> Dataset:
    """Load HuggingFace dataset with volume-based caching.

    Loads specified splits from HuggingFace, utilizing cached versions when
    available. Concatenates multiple splits and supports sampling.

    Args:
        dataset_name: HuggingFace dataset identifier
        splits: Split names to load (e.g., ['train', 'test'])
        n_samples: Optional sample limit per split
        seed: Random seed for shuffling
        cache_dir: Root directory for persistent cache

    Returns:
        Combined and optionally sampled dataset
    """
    cache_root = Path(cache_dir) / dataset_name.replace("/", "_")
    cache_root.mkdir(parents=True, exist_ok=True)

    loaded_splits: list[Dataset] = []

    for split_name in splits:
        split_cache_dir = cache_root / split_name

        ds = _load_split_from_cache(split_cache_dir, dataset_name, split_name)

        if ds is None:
            ds = _download_split(dataset_name, split_name)
            _cache_dataset_split(ds, split_cache_dir)

        loaded_splits.append(ds)

    if not loaded_splits:
        raise ValueError("No splits provided to load the dataset")

    combined_ds = concatenate_datasets(loaded_splits)

    logger.info("Shuffling dataset with seed=%d", seed)
    combined_ds = combined_ds.shuffle(seed=seed)

    if n_samples is not None:
        actual_samples = min(n_samples, combined_ds.num_rows)
        combined_ds = combined_ds.select(range(actual_samples))

    logger.info("Dataset loaded: %s with %d rows", dataset_name, combined_ds.num_rows)

    return combined_ds


def _authenticate_huggingface() -> str | None:
    """Authenticate with HuggingFace Hub using environment token."""
    token = os.getenv("HF_TOKEN")
    if token:
        logger.info("Authenticating with HuggingFace Hub")
        login(token=token)
    else:
        logger.warning("No HF_TOKEN found in environment")
    return token


def _load_cached_model(
    processor_path: Path, model_path: Path
) -> tuple[AutoProcessor, AutoModelForImageTextToText] | None:
    """Load model and processor from cached paths."""
    if not (processor_path.exists() and model_path.exists()):
        return None

    logger.info("Loading cached model from: %s", model_path.parent)

    try:
        _apply_config_fix(str(model_path))
    except Exception as fix_error:
        logger.warning("Could not apply config fix: %s", fix_error)

    try:
        proc = AutoProcessor.from_pretrained(
            str(processor_path),
            max_image_tokens=256,
            local_files_only=True,
        )

        mdl = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            torch_dtype="bfloat16",
            device_map="auto",
            local_files_only=True,
        )

        logger.info("Successfully loaded model from cache")
        return proc, mdl

    except Exception as load_error:
        logger.warning("Failed to load from cache: %s", load_error)
        return None


def _download_and_persist_model(
    model_id: str,
    auth_token: str | None,
    processor_path: Path,
    model_path: Path,
) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    """Download model from HuggingFace and cache locally."""
    logger.info("Downloading model: %s", model_id)

    proc = AutoProcessor.from_pretrained(
        model_id,
        max_image_tokens=256,
        token=auth_token,
    )

    mdl = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype="bfloat16",
        device_map="auto",
        token=auth_token,
    )

    try:
        logger.info("Persisting processor to: %s", processor_path)
        proc.save_pretrained(str(processor_path))
        logger.debug("Processor saved")
    except Exception as save_error:
        logger.warning("Failed to save processor: %s", save_error)

    try:
        logger.info("Persisting model to: %s", model_path)
        mdl.save_pretrained(str(model_path))

        try:
            _apply_config_fix(str(model_path))
        except Exception as fix_error:
            logger.warning("Could not apply config fix: %s", fix_error)

        logger.debug("Model saved")
    except Exception as save_error:
        logger.warning("Failed to save model: %s", save_error)

    return proc, mdl


def load_model_and_processor(
    model_id: str,
    cache_dir: str = "/models",
) -> tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load VLM model and processor with volume-based caching.

    Retrieves model artifacts from HuggingFace with intelligent caching to
    Modal volumes. Handles authentication and applies necessary config patches.

    Args:
        model_id: HuggingFace model identifier
        cache_dir: Root directory for model cache

    Returns:
        Tuple of (model, processor)
    """
    cache_root = Path(cache_dir) / model_id.replace("/", "_")
    cache_root.mkdir(parents=True, exist_ok=True)

    processor_cache = cache_root / "processor"
    model_cache = cache_root / "model"

    auth_token = _authenticate_huggingface()

    cached_result = _load_cached_model(processor_cache, model_cache)

    if cached_result is not None:
        proc, mdl = cached_result
    else:
        logger.info("Cache miss, downloading from HuggingFace")
        proc, mdl = _download_and_persist_model(
            model_id, auth_token, processor_cache, model_cache
        )

    vocab_count = len(proc.tokenizer)
    param_count = mdl.num_parameters()
    size_gb = param_count * 2 / 1e9

    logger.info("Model loaded successfully")
    logger.info("Vocabulary size: %d", vocab_count)
    logger.info("Parameter count: %s", f"{param_count:,}")
    logger.info("Model size (bfloat16): %.1f GB", size_gb)

    return mdl, proc
