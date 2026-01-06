"""Training callbacks for model checkpoint management.

Provides custom callback implementations for HuggingFace Trainer to handle
auxiliary artifacts like processors and tokenizers during checkpoint saves.
"""

import logging
import os

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class ProcessorSaveCallback(TrainerCallback):
    """Persist processor artifacts alongside model checkpoints.
    
    Ensures that tokenizers and image processors are saved during training
    to maintain complete checkpoint reproducibility.
    """

    def __init__(self, processor):
        """Initialize callback with processor reference.

        Args:
            processor: Processor instance (tokenizer + image processor) to persist
        """
        self.processor = processor

    def on_save(self, args, state, control, **kwargs):
        """Hook executed when trainer saves a checkpoint.
        
        Saves the processor to the checkpoint directory for complete artifact
        preservation.

        Args:
            args: Training arguments containing output directory
            state: Current training state with step counter
            control: Training control flow object
            **kwargs: Additional callback arguments
        """
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        logger.info("Persisting processor to: %s", checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        logger.debug("Processor saved successfully")
