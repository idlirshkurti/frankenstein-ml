set shell := ["powershell.exe", "-c"]

# Evaluate model with specified config
evaluate config:
    uv run modal run -m src.car_maker_identification.evaluate --config-file-name {{config}}

# Open evaluation visualization notebook
report:
    uv run jupyter notebook notebooks/visualize_evals.ipynb

# Fine-tune model with specified config
fine-tune config:
    uv run modal run -m src.car_maker_identification.fine_tune --config-file-name {{config}}

# Run linter with auto-fix
lint:
    uv run ruff check --fix .

# Format code
format:
    uv run ruff format .

# Run both linting and formatting
code-fixes: lint format
