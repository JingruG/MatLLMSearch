# README 

This code implements an evolutionary search pipeline for crystal structure generation (CSG) and crystal structure prediction (CSP) with Large Language Models (LLMs) without fine-tuning.

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

## Prerequisites

### Required Python Packages

install key packages in `requirements.txt`

```bash
conda create -n matllmsearch python=3.10
conda activate matllmsearch
conda install -c conda-forge mamba
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```


### External Dependencies

1. Meta-Llama 3.1

2. MatBench Dataset [`matbench_v0.1 matbench_expt_gap`](https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_gap/)

   Download known stable structures with decomposition energy: [Seed Structures](https://drive.google.com/file/d/1DqE9wo6dqw3aSLEfBx-_QOdqmtqCqYQ5/view?usp=sharing)

3. `mp_patched_phase_diagram`:  [`PatchedPhaseDiagram`](https://github.com/materialsproject/pymatgen/blob/v2023.5.10/pymatgen/analysis/phase_diagram.py#L1480-L1814) constructed from all MP `pymatgen` `ComputedStructureEntries`.

   Download [oracle/resouorces/2023-02-07-ppd-mp.pkl.gz](https://figshare.com/ndownloader/files/48241624).

4. CHGNet model

## Usage

### CSG

```bash
python main.py --task csg --opt_goal e_hull_distance --max_iter 10
```

### CSP

For crystal structure prediction of Na3AlCl6:

```bash
python main.py --task csp --opt_goal e_hull_distance --max_iter 10
```

For crystal structure prediction of Ag6O2:

```bash
python main.py --task csp --opt_goal e_hull_distance --max_iter 10 --csp_compound "Ag6O2"
```
