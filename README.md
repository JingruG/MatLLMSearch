# MatLLMSearch: Crystal Structure Discovery with Evolution-Guided Large Language Models

This is the implementation for **MatLLMSearch: Crystal Structure Discovery with Evolution-Guided Large Language Models**. This code implements an evolutionary search pipeline for crystal structure generation (CSG) with Large Language Models (LLMs) without fine-tuning.

## Pipeline Overview

<div align="center">
  <img src="assets/pipeline.png" alt="main_pipeline">
</div>

### How Pareto Frontiers are pushed during Iterations under Varied Optimization Objectives

<div align="center">
  <img src="assets/pareto_evolution.gif" alt="pareto_evolution" loop>
</div>

### Crystal Structure Prediction Examples
<div align="center">
  <img src="assets/crystal_structures_Ag6O2.gif" alt="Ag6O2_csp_examples" width="250">
  <img src="assets/crystal_structures_Bi2F8.gif" alt="Bi2F8_csp_examples" width="250">
  <img src="assets/crystal_structures_Co4B2.gif" alt="Co4B2_csp_examples" width="250">
</div>

<div align="center" style="margin-top: 10px;">
  <img src="assets/crystal_structures_KZnF3.gif" alt="KZnF3_csp_examples" width="250">
  <img src="assets/crystal_structures_Sr2O4.gif" alt="Sr2O4_csp_examples" width="250">
  <img src="assets/crystal_structures_YMg3.gif" alt="YMg3_csp_examples" width="250">
</div>


## Installation

1. Install MatLLMSearch dependencies:
```bash
pip install -r requirements.txt
```

2. Configure models and credentials:
   - Copy `config/credentials.yaml` and add your API keys
   - Configure models in `config/models.yaml` (already includes common models)

3. Download required data files:
```bash
# Create data directory
mkdir -p data

# Download seed structures (optional - enables few-shot generation)
# You may download data/band_gap_processed.csv at https://drive.google.com/file/d/1DqE9wo6dqw3aSLEfBx-_QOdqmtqCqYQ5/view?usp=sharing
# Or data/band_gap_processed_5000.csv at https://drive.google.com/file/d/14e5p3EoKzOHqw7hKy8oDsaGPK6gwhnLV/view?usp=sharing

# Download phase diagram data (required for E_hull distance calculations)
wget -O data/2023-02-07-ppd-mp.pkl.gz https://figshare.com/ndownloader/files/48241624
```

**Note**: 
- All configuration is managed through local `config/` directory
- Models are configured in `config/models.yaml`
- API keys are configured in `config/credentials.yaml` 

## Quick Start

### Crystal Structure Generation (CSG)
Generate novel crystal structures using evolutionary optimization:

```bash
python cli.py csg \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --population-size 100 \
    --max-iter 10 \
    --opt-goal e_hull_distance \
    --data-path data/band_gap_processed.csv \
    --save-label csg_experiment
```

### Crystal Structure Prediction (CSP)
Predict ground state structures for a target compound:

```bash
python cli.py csp \
    --compound Ag6O2 \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --population-size 10 \
    --max-iter 5 \
    --save-label ag6o2_prediction
```

### Analysis
The `analyze` command evaluates generated structures and computes comprehensive metrics including:
- Structural validity and composition validity
- Structural diversity and composition diversity
- Structural novelty and composition novelty (vs reference pool)
- Overall novelty (fraction of structures that are both compositionally and structurally novel)
- M3GNet metastability
- Stability rates (CHGNet)

#### Evaluate Existing Results

**Option 1: From a CSV file**
```bash
python cli.py analyze \
    --input data/llama_test.csv \
    --output evaluation_results.json \
    --data-path data/band_gap_processed.csv
```

**Option 2: From a previous CSG run directory**
```bash
python cli.py analyze \
    --results-path logs/analyze_generation \
    --output reevaluated_results.json \
    --data-path data/band_gap_processed_5000.csv
```
This will look for `generations.csv` in the specified results path.

#### Generate and Evaluate via API

Generate structures using the CSG evolutionary workflow with API models and then evaluate:

```bash
python cli.py analyze --generate \
    --model openai/gpt-5-mini \
    --data-path data/band_gap_processed_5000.csv \
    --max-iter 10 \
    --population-size 10 \
    --reproduction-size 5 \
    --parent-size 2 \
    --output gpt5_results.json
```

**Key parameters for API generation:**
- `--generate`: Flag to enable API generation (uses CSG workflow)
- `--model`: Model to use (e.g., `openai/gpt-5-mini`, `openai/gpt-4o-mini`)
- `--data-path`: Path to seed structures CSV (used as reference pool for novelty)
- `--max-iter`: Number of evolutionary iterations
- `--population-size`: Initial population size
- `--reproduction-size`: Number of offspring per generation
- `--parent-size`: Number of parent structures per group

**Note:** All generated structures are kept and deduplicated after all iterations complete before evaluation.

#### Finding Results from Previous Runs

When you run `analyze --generate`, the CSG workflow saves intermediate results to:
- `logs/analyze_generation/generations.csv`: All generated structures with properties
- `logs/analyze_generation/metrics.csv`: Per-iteration metrics

The final evaluation summary is saved to the `--output` file you specify (e.g., `gpt5_results.json`).

To re-evaluate a previous run:
```bash
python cli.py analyze \
    --results-path logs/analyze_generation \
    --output new_evaluation.json \
    --data-path data/band_gap_processed_5000.csv
```

## Configuration Options

### Models
MatLLMSearch uses a unified model interface with support for local models or API.
Model configuration is handled via `config/models.yaml` and `config/credentials.yaml` files.

### Optimization Goals
- `e_hull_distance`: Minimize energy above convex hull (stability)
- `bulk_modulus_relaxed`: Maximize bulk modulus (mechanical properties)
- `multi-obj`: Multi-objective optimization combining both

### Structure Formats
- `poscar`: VASP POSCAR format
- `cif`: Crystallographic Information File format
