"""Modal infrastructure configuration utilities.

Provides helper functions for setting up Modal compute resources including
applications, Docker images, persistent volumes, retry policies, and secrets.
Used to configure serverless GPU environments for model training and evaluation.
"""

import modal


def get_modal_app(name: str) -> modal.App:
    """Create Modal application instance with specified name."""
    return modal.App(name)


def get_docker_image() -> modal.Image:
    """Build Modal container image with required ML dependencies.
    
    Configures a Debian slim base image with Python 3.12 and installs
    all necessary packages for VLM training and inference workflows.
    """
    base_image = modal.Image.debian_slim(python_version="3.12")
    
    dependencies = [
        "datasets>=4.1.1",
        "modal>=1.1.4",
        "outlines>=1.2.7",
        "peft>=0.15.2",
        "pydantic-settings>=2.10.1",
        "tqdm>=4.67.1",
        "transformers==4.57.1",
        "trl==0.24.0",
        "pillow>=11.3.0",
        "matplotlib>=3.10.6",
        "torchao>=0.4.0",
        "wandb>=0.22.2",
        "torchvision==0.23.0",
        "bitsandbytes",
        "seaborn",
        "scikit-learn",
    ]
    
    configured_image = base_image.uv_pip_install(*dependencies).env(
        {"HF_HOME": "/model_cache"}
    )
    
    return configured_image


def get_volume(name: str) -> modal.Volume:
    """Retrieve or create named persistent volume for data storage."""
    return modal.Volume.from_name(name, create_if_missing=True)


def get_retries(max_retries: int) -> modal.Retries:
    """Configure retry behavior for task failures.
    
    Args:
        max_retries: Maximum number of retry attempts
        
    Returns:
        Retry policy with no initial delay
    """
    return modal.Retries(initial_delay=0.0, max_retries=max_retries)


def get_secrets() -> list[modal.Secret]:
    """Load required secrets for remote execution.
    
    Returns:
        List containing Weights & Biases authentication secret
    """
    wb_secret = modal.Secret.from_name("wandb-secret")
    return [wb_secret]
