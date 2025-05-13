import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from pathlib import Path
from bs4 import BeautifulSoup
import re
from basic_eval import timeout, TimeoutError
import json
from utils.config import METAL_OXIDATION_STATES
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core.structure import Structure
from utils.generator import GenerationResult
from collections import Counter
from dataclasses import dataclass

import importlib
import utils.basic_eval
importlib.reload(utils.basic_eval)
from utils.basic_eval import structure_to_crystal, CDVAEGenEval, CompScaler, timeout, TimeoutError


        

def has_MnO2(composition):
    try:
        comp = composition.as_dict()
        if 'Mn' not in comp or 'O' not in comp:
            return False
        return comp['Mn']/comp['O'] == 0.5
    except:
        return False


def generation_dedup(file_path) :
    df = pd.read_csv(file_path)
    unique_structures = []
    keep_indices = []
    reduced_formulas = []
    for idx, struct_str in enumerate(df['StructureRelaxed']):
        try:
            structure = Structure.from_dict(json.loads(struct_str))
            reduced_formula = structure.composition.reduced_formula
            df.at[idx, 'Composition'] = structure.composition
            df.at[idx, 'composition_str'] = structure.composition.formula
            is_duplicate = any(
                structure.matches(unique_struct, scale=True, attempt_supercell=False)
                for unique_struct, formula in zip(unique_structures, reduced_formulas)
                if formula == reduced_formula
            )
            if not is_duplicate:
                unique_structures.append(structure)
                reduced_formulas.append(reduced_formula)
                keep_indices.append(idx)
        except Exception as e:
            print(f"Error processing structure at index {idx}: {e}")
            continue
    return df.iloc[keep_indices].reset_index(drop=True)


def valid_value(x):
    return (x is not None and
            not np.isinf(x) and
            not np.isnan(x) and
            x != 0)


def valid_mean(values):
    valid_values = [x for x in values if valid_value(x)]
    return np.mean(valid_values) if valid_values else 0.0



def calculate_cumulative_metrics(df, iter_num):
    iter_df = df[df['Iteration'] <= iter_num]
    total = len(iter_df)
    valid_ehull = iter_df['EHullDistance'].dropna()
    valid_bulk = iter_df['BulkModulusRelaxed'].dropna()[lambda x: x > 0] * conversion_factor
    valid_deltae = iter_df['DeltaE'].dropna()
    non_f_ele = len(iter_df[~iter_df['composition_str'].str.contains('|'.join(LANTHANIDES + ACTINIDES), na=False)])
    stable_10 = len(iter_df[iter_df['EHullDistance'] < 0.1])
    stable_03 = len(iter_df[iter_df['EHullDistance'] < 0.03])
    return {
        'iteration': iter_num,
        'total': total,
        'stable_10': stable_10,
        'stable_03': stable_03,
        'stable_10_ratio': stable_10 / total if total > 0 else 0,
        'stable_03_ratio': stable_03 / total if total > 0 else 0,
        'avg_ehull': valid_mean(valid_ehull),
        'max_ehull': max(valid_ehull),
        'min_ehull': min(valid_ehull),
        'avg_bulk': valid_mean(valid_bulk),
        'avg_deltae': valid_mean(valid_deltae),
        'non_f_ele': non_f_ele,
        'non_f_ele_ratio': non_f_ele / total if total > 0 else 0
    }
    ratio_order = {'1_5': 0, '2_5': 1, '2_2': 2, '5_5': 3, '5_2': 4}


def create_gif_from_pngs(input_folder, output_gif_path, duration=50, pattern='*.png', sort_key=None):
    png_files = sorted(glob.glob(os.path.join(input_folder, pattern)), key=sort_key)
    if not png_files:
        raise ValueError(f"No PNG files found in {input_folder} matching pattern {pattern}")
    with Image.open(png_files[0]) as first_img:
        target_size = first_img.size
    images = []
    for file in png_files:
        with Image.open(file) as img:
            if img.size != target_size:
                img = img.resize(target_size, Image.Resampling.LANCZOS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(np.array(img))
    imageio.mimsave(output_gif_path, images, duration=duration, loop=0)


if __name__ == "__main__":
save_label = "poscar_deepseek_r3_100_2_5_3500"
generation_file = f'../results/{save_label}/generations.csv'
metric_file = f'../results/{save_label}/metrics.csv'
generations_df = pd.read_csv(generation_file)
metrics_df = pd.read_csv(metric_file)
total_raw = metrics_df['num_a'].sum()
print('Total raw structures output:', total_raw)
print('Total structures:', len(generations_df))
generations_df = generation_dedup(generation_file)
print('Total unique structures:', len(generations_df))

seed_structures_df = pd.read_csv('seed_structures_processed_3500.csv')
seed_structures_df['structure'] = seed_structures_df['structure'].apply(lambda x: Structure.from_str(x, fmt='json') if pd.notna(x) else None)
seed_structures_df['composition'] = [s.composition for s in seed_structures_df['structure']]
seed_structures_df['composition_str'] = [s.composition.formula for s in seed_structures_df['structure']]
extra_pool_df = seed_structures_df[np.isfinite(seed_structures_df['e_hull_distance'])]
len(extra_pool_df)



generations_df['structure'] = generations_df['StructureRelaxed'].apply(lambda x: Structure.from_str(x, fmt='json') if pd.notna(x) else None)
generations_df['crys'] = generations_df['structure'].apply(lambda x: structure_to_crystal(x) if pd.notna(x) else None)
pred_crys = generations_df['crys'].tolist()
gt_novelty_crys = extra_pool_df[np.isfinite(extra_pool_df['e_hull_distance'])]['structure'].tolist()
gt_novelty_crys = [structure_to_crystal(x) for x in gt_novelty_crys]


settings = {
    # 'ex_f_ele': 'poscar_p8_70b_100_2_5_ehull_non_f_ele',
    # 'bulk': 'poscar_p8_70b_100_2_5_bulk',
    # 'multi_obj': 'poscar_p8_70b_100_2_5_ehull_bulk',
    # 'multi_obj_2': 'poscar_p8_70b_100_2_5_multi_obj_2_3578',
    'ehull': 'poscar_p8_70b_100_2_5_ehull_3500_rerun',
    'random_init': 'poscar_p8_70b_100_2_5_random_init',
    'pool1000': 'poscar_p8_70b_100_2_5_ehull_pool1000',
    'deepseek': 'poscar_deepseek_r3_100_2_5_3500',
}
results = {}
for name, label in settings.items():
    file_path = f'../results/{label}/generations.csv'
    generations_df = generation_dedup(file_path)
    generations_df['structure'] = generations_df['StructureRelaxed'].apply(lambda x: Structure.from_str(x, fmt='json') if pd.notna(x) else None)
    generations_df['crys'] = generations_df['structure'].apply(lambda x: structure_to_crystal(x) if pd.notna(x) else None)
    results[name] = generations_df['crys'].tolist()

for name, pred_crys in results.items():
    print(name)
    metrics = CDVAEGenEval(
            pred_crys, 
            gt_novelty_crys,
            gt_novelty_crys,
            n_samples=len(pred_crys), 
            eval_model_name='mp20'
        ).get_metrics()
