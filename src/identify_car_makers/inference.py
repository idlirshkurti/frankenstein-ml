"""Model inference utilities for vision-language tasks.

Provides functions for running inference on VLM models with support for both
structured outputs (using Outlines library) and unstructured text generation.
Handles single-sample and batch processing modes.
"""

import logging

from outlines import Model, from_transformers
from outlines.inputs import Chat, Image as OutlinesImage
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .output_types import CarIdentificationOutputType

logger = logging.getLogger(__name__)


def _construct_chat_prompt(system_msg: str, user_msg: str, img: Image.Image) -> Chat:
    """Construct chat prompt from system message, user message, and image."""
    return Chat(
        [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": OutlinesImage(img)},
                    {"type": "text", "text": user_msg},
                ],
            },
        ]
    )


def _parse_single_response(raw_output: str, index: int = 0) -> CarIdentificationOutputType | None:
    """Parse raw model output into structured type."""
    try:
        return CarIdentificationOutputType.model_validate_json(raw_output)
    except Exception as parse_error:
        logger.error("Failed to parse response %d: %s", index, parse_error)
        logger.debug("Unparseable output: %s", raw_output)
        return None


def _process_single_image(
    vlm: Model,
    sys_prompt: str,
    usr_prompt: str,
    img: Image.Image,
    token_limit: int | None,
) -> CarIdentificationOutputType | None:
    """Run inference on a single image with structured output."""
    chat_prompt = _construct_chat_prompt(sys_prompt, usr_prompt, img)
    raw_result: str = vlm(chat_prompt, CarIdentificationOutputType, max_new_tokens=token_limit)
    return _parse_single_response(raw_result)


def _process_batch_images(
    vlm: Model,
    sys_prompt: str,
    usr_prompt: str,
    imgs: list[Image.Image],
    token_limit: int | None,
) -> list[CarIdentificationOutputType | None]:
    """Run batched inference on multiple images with structured output."""
    prompt_batch = [_construct_chat_prompt(sys_prompt, usr_prompt, img) for img in imgs]
    
    try:
        raw_results: list[str] = vlm.batch(
            prompt_batch,
            output_type=CarIdentificationOutputType,
            max_new_tokens=token_limit,
        )
        return [_parse_single_response(result, idx) for idx, result in enumerate(raw_results)]
    except Exception as batch_error:
        logger.error("Batch inference failed: %s", batch_error)
        return [None] * len(imgs)


def get_structured_model_output(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    system_prompt: str,
    user_prompt: str,
    images: Image.Image | list[Image.Image],
    max_new_tokens: int | None = 64,
) -> CarIdentificationOutputType | list[CarIdentificationOutputType] | None:
    """Execute structured inference on single or multiple images.

    Wraps the model with Outlines to enforce structured JSON output conforming
    to CarIdentificationOutputType schema.

    Args:
        model: Vision-language model for inference
        processor: Tokenizer/processor for the model
        system_prompt: System-level instruction
        user_prompt: User query text
        images: Single image or batch of images
        max_new_tokens: Generation length limit

    Returns:
        Structured output(s) or None on error
    """
    vlm = from_transformers(model, processor)
    
    is_single = isinstance(images, Image.Image)
    
    if is_single:
        return _process_single_image(vlm, system_prompt, user_prompt, images, max_new_tokens)
    else:
        return _process_batch_images(vlm, system_prompt, user_prompt, images, max_new_tokens)


def get_structured_model_output_batch(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    system_prompt: str,
    user_prompt: str,
    images: list[Image.Image],
    max_new_tokens: int | None = 64,
) -> list[CarIdentificationOutputType | None]:
    """Batch inference wrapper for structured outputs.

    Convenience function that delegates to get_structured_model_output
    with explicit batch semantics.

    Args:
        model: Vision-language model
        processor: Model processor
        system_prompt: System instruction
        user_prompt: User query
        images: List of images to process
        max_new_tokens: Token generation limit

    Returns:
        List of structured predictions
    """
    return get_structured_model_output(
        model, processor, system_prompt, user_prompt, images, max_new_tokens
    )


def get_model_output(
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    conversation: list[dict],
    max_new_tokens: int | None = 64,
) -> str:
    """Generate unstructured text output from conversation history.

    Runs standard autoregressive generation without output constraints.

    Args:
        model: Vision-language model
        processor: Model processor/tokenizer
        conversation: Chat history with roles and content
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text response
    """
    model_inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        tokenize=True,
    ).to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    
    response_only = generated_ids[:, model_inputs["input_ids"].shape[1] :]
    decoded_text = processor.batch_decode(response_only, skip_special_tokens=True)[0]
    
    return decoded_text