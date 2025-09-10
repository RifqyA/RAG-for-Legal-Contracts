# RAG Pipeline

This repository converts the original Jupyter notebook `RAG_PIPELINE_MAIN.ipynb` into a reproducible Python package.

## What changed
- Consolidated imports into one section.
- Removed duplicate cells and repeated function/class definitions (listed below).
- Removed notebook magics (`%...`, `!pip ...`) and kept pure Python.
- Produced a single entry module: `rag_pipeline/pipeline.py`.

## Redundancies removed
- Duplicate cells (by content hash): 0
- Duplicate function/class definitions removed: 0

See `artifacts/redundancy_report.json` for details.

## Quick start

```bash
# 1) Create and activate env (example with venv)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install this project in editable mode
pip install -e .

# 3) Run the pipeline (module entry)
python -m rag_pipeline.pipeline
```

> Note: If the original notebook expected data paths or environment variables, set them in `.env` (see `.env.example`).

## Project layout
```
rag-pipeline/
├─ src/
│  └─ rag_pipeline/
│     └─ pipeline.py
├─ tests/
├─ artifacts/
│  └─ redundancy_report.json
├─ pyproject.toml
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## Requirements
`requirements.txt` was inferred from import statements in the notebook. You may need to add or remove items based on your environment.

## How to publish to GitHub (new repo)

```bash
# Configure Git (if you haven't)
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# Initialize and commit
git init
git add .
git commit -m "Initial commit: RAG pipeline (from notebook)"

# Create a new repo on GitHub first (via UI).
# Then add the remote and push:
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## How to push changes later
```bash
git add -A
git commit -m "Update pipeline / fix X"
git push
```

