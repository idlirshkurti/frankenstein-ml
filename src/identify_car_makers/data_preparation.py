"""Dataset preprocessing utilities for training workflows.

Provides functions for splitting datasets and transforming raw samples into
conversation-formatted structures suitable for supervised fine-tuning of
vision-language models.
"""

from typing import Any

from datasets import Dataset


def split_dataset(
    dataset: Dataset,
    test_size: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Partition dataset into training and evaluation subsets.

    Args:
        dataset: Source dataset to split
        test_size: Proportion of data for evaluation (0 < test_size < 1)
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_dataset, eval_dataset)

    Raises:
        ValueError: If test_size is not in valid range
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    return splits["train"], splits["test"]


def _construct_conversation(
    img_sample,
    lbl_text: str,
    sys_instruction: str,
    usr_query: str,
) -> list[dict[str, list[dict[str, Any]]]]:
    """Build conversation structure for a single training sample."""
    return [
        {"role": "system", "content": [{"type": "text", "text": sys_instruction}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_sample},
                {"type": "text", "text": usr_query},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": lbl_text}],
        },
    ]


def format_dataset_as_conversation(
    dataset: Dataset,
    system_prompt: str,
    user_prompt: str,
    image_column: str,
    label_column: str,
    label_mapping: dict[Any, str] | None,
) -> list[list[dict]]:
    """Transform dataset into conversation format for SFT training.

    Converts each dataset sample into a three-turn conversation structure
    (system, user, assistant) suitable for supervised fine-tuning.

    Args:
        dataset: Source dataset with image and label columns
        system_prompt: System instruction for conversation
        user_prompt: User query template
        image_column: Name of column containing images
        label_column: Name of column containing labels
        label_mapping: Optional mapping from raw labels to target strings

    Returns:
        List of conversation-formatted samples
    """

    def transform_sample(raw_sample):
        label_value = raw_sample[label_column]
        label_text = label_mapping[label_value] if label_mapping else label_value
        
        return _construct_conversation(
            raw_sample[image_column],
            label_text,
            system_prompt,
            user_prompt,
        )

    return [transform_sample(sample) for sample in dataset]
