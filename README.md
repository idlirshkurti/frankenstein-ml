# Car Maker Identification

A vision-language model (VLM) project for identifying car manufacturers from images. This repo lets you evaluate and fine-tune state-of-the-art VLMs on car classification tasks using serverless GPU infrastructure.

## What's This About?

Ever wondered how well modern vision-language models can recognize car brands? This project helps you find out. It's built around the Stanford Cars dataset and uses models like LiquidAI's LFM2-VL series to classify car images by manufacturer.

The cool part? Everything runs on serverless GPUs via [Modal](https://modal.com/), so you don't need expensive hardware sitting around. And with [Weights & Biases](https://wandb.ai/) integration, you get beautiful experiment tracking right out of the box.

## Features

- **Evaluation Pipeline**: Test any compatible VLM on car identification tasks
- **Fine-Tuning Support**: Use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
- **Structured Generation**: Enforce valid outputs using the Outlines library
- **Batch Processing**: Efficient batched inference for faster evaluation
- **Experiment Tracking**: Built-in W&B integration for metrics and visualizations
- **Serverless Compute**: No local GPU needed - runs on Modal's infrastructure

## Getting Started

### Prerequisites

- Python 3.13
- [UV](https://docs.astral.sh/uv/) package manager
- [Modal](https://modal.com/) account (for running on GPUs)
- [Weights & Biases](https://wandb.ai/) account (optional, for experiment tracking)

### Installation

We use UV for dependency management. If you don't have it yet:

```bash
# On Windows with scoop
scoop install uv

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and set up the environment:

```bash
git clone https://github.com/yourusername/identify_car_makers.git
cd identify_car_makers

# Create virtual environment with Python 3.13
uv venv --python 3.13

# Install dependencies
uv sync
```

### Quick Start

The project uses [Just](https://github.com/casey/just) as a command runner. Install it with:

```bash
scoop install just
```

## Usage

### Evaluating a Model

Run an evaluation using a config file:

```bash
just evaluate configs/eval_lfm_450M.yaml
```

This will:
1. Download the model and dataset (cached for reuse)
2. Run inference on the test set
3. Generate accuracy metrics and confusion matrices
4. Log everything to Weights & Biases

### Fine-Tuning

Fine-tune a model with LoRA adapters:

```bash
just fine-tune configs/train_config.yaml
```

The training process:
- Splits your dataset into train/eval sets
- Applies LoRA adapters to the model
- Trains with gradient accumulation and mixed precision
- Saves checkpoints to Modal volumes
- Tracks metrics in W&B

### Viewing Results

Check out evaluation results:

```bash
just report
```

This opens a Jupyter notebook with visualizations of your predictions, including confusion matrices and image galleries of correct/incorrect classifications.

## Project Structure

```
identify_car_makers/
├── src/identify_car_makers/
│   ├── artifacts.py          # Model and dataset loading with caching
│   ├── batching.py           # Batch creation utilities
│   ├── callbacks.py          # Training callbacks
│   ├── config.py             # Configuration schemas
│   ├── data_preparation.py   # Dataset preprocessing
│   ├── evaluate.py           # Evaluation pipeline
│   ├── fine_tune.py          # Training pipeline
│   ├── inference.py          # Model inference utilities
│   ├── modal_infra.py        # Modal infrastructure setup
│   ├── output_types.py       # Structured output schemas
│   ├── peft.py               # LoRA configuration
│   └── report.py             # Results visualization
├── configs/                   # YAML config files
├── evals/                     # Evaluation results (CSV)
├── justfile                   # Task automation
└── pyproject.toml            # Project dependencies
```

## Configuration

Configs live in `configs/` as YAML files. Here's what a typical eval config looks like:

```yaml
seed: 23
model: LiquidAI/LFM2-VL-450M
structured_generation: true

dataset: Paulescu/stanford_cars
n_samples: 100
split: test

system_prompt: |
  You excel at identifying car makers from pictures.

user_prompt: |
  What car maker do you see in this picture?
  Pick one from the following list: ...

image_column: image
label_column: maker
```

## Development

Format and lint your code:

```bash
just code-fixes
```

This runs both Ruff linting and formatting to keep the codebase clean.

## How It Works

1. **Data Loading**: Datasets are downloaded from HuggingFace and cached to Modal volumes for fast reuse
2. **Model Loading**: Models are loaded with bfloat16 precision and cached similarly
3. **Inference**: Images are processed through the VLM with either structured (schema-enforced) or free-form generation
4. **Evaluation**: Predictions are compared against ground truth labels, generating accuracy metrics and confusion matrices

The structured generation mode uses the Outlines library to guarantee valid JSON outputs matching the car maker schema - no more parsing headaches!

## Why This Project?

I built this to experiment with vision-language models on a fun, practical task. Car identification is challenging enough to be interesting but concrete enough to evaluate objectively. Plus, it's a great testbed for trying out new VLMs, fine-tuning techniques, and MLOps patterns.

The serverless approach via Modal means you can run serious GPU workloads without maintaining infrastructure. And the UV + Just combo keeps dependency management and task running painless.

## Contributing

Found a bug? Want to add support for a new model or dataset? PRs welcome! Just make sure to run `just code-fixes` before committing.

## License

MIT - feel free to use this however you'd like.

## Acknowledgments

- Stanford Cars dataset from [Paulescu](https://huggingface.co/datasets/Paulescu/stanford_cars)
- LiquidAI for their awesome LFM2-VL models
- Modal for making serverless GPU compute accessible
- The HuggingFace team for transformers, datasets, and PEFT libraries
