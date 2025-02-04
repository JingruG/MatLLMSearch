# MatLLMSearch: Large Language Models Are Innate Crystal Structure Generators

This is the implementation for **MatLLMSearch: Large Language Models Are Innate Crystal Structure Generators**. This code implements an evolutionary search pipeline for crystal structure generation (CSG) and crystal structure prediction (CSP) with Large Language Models (LLMs) without fine-tuning.

## Pipeline Overview

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1a26VlO27v8mK3P2jGv7XX3ZHKmzPpmUk" alt="main_pipeline" loop>
</div>

### Elemental Frequency in LLM-proposed Crystal Structures (Periodic Table)

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1-Ex-mNduWgRSfPDooW89OT1snN2EXhc7" alt="periodic_animation" loop>
</div>

### How Pareto Frontiers are pushed during Iterations under Varied Optimization Objectives

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1-KWyKDwJRVH7mB-5sXZTNvwglYHJEcKq" alt="pareto_evolution" loop>
</div>

## Prerequisites

### Required Python Packages

install key packages in `requirements.txt`

### External Dependencies

1. Meta-Llama 3.1

2. MatBench Dataset [`matbench_v0.1 matbench_expt_gap`](https://matbench.materialsproject.org/Leaderboards%20Per-Task/matbench_v0.1_matbench_expt_gap/)

   Download `Extra Pool` of known stable structures with decomposition energy: [3.5K Extra Pool](https://drive.google.com/file/d/116ielqvZJAn2M4oKTnz2snbHArgHsDcA/view?usp=sharing)

3. `mp_patched_phase_diagram`:  [`PatchedPhaseDiagram`](https://github.com/materialsproject/pymatgen/blob/v2023.5.10/pymatgen/analysis/phase_diagram.py#L1480-L1814) constructed from all MP `pymatgen` `ComputedStructureEntries`.

   Download [oracle/2023-02-07-ppd-mp.pkl.gz](https://figshare.com/ndownloader/files/48241624).

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

## Citation

If you use MatLLMSearch, please cite our paper:

```
@article{gan2025matllmsearch,
  title={Large Language Models Are Innate Crystal Structure Generators},
  author={Gan, Jingru and Zhong, Peichen and Du, Yuanqi and Zhu, Yanqiao and 
          Duan, Chenru and Wang, Haorui and Gomes, Carla P. and Persson, Kristin and 
          Schwalbe-Koda, Daniel and Wang, Wei},
  journal={arXiv preprint arXiv:},
  year={2025}
}
```