import argparse
from pathlib import Path
import pandas as pd
import torch
import json
import random
import numpy as np
from typing import List, Tuple
import tiktoken
from dataclasses import dataclass
import os
import time
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

from utils.config import LANTHANIDES, ACTINIDES
from utils.llm_manager import LLMManager
from utils.generator import StructureGenerator, GenerationResult
from utils.evaluator import StructureEvaluator
from utils.stability import StabilityCalculator
from typing import List, Dict, Tuple, Any

from collections import Counter
import torch.multiprocessing as mp



def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    
    # Task and Model arguments
    parser.add_argument('--base_model', default='meta-llama/Meta-Llama-3.1-70B-Instruct')
    parser.add_argument('--model_label', default='llama3_instruct')
    parser.add_argument('--fmt', choices=['poscar', 'cif'], default='poscar')
    parser.add_argument('--tensor_parallel_size', type=int, default=4)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.84)
    parser.add_argument('--max_tokens', type=int, default=4000)
    parser.add_argument('--temperature', type=float, default=1.0)
    
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--reproduction_size', type=int, default=5)
    parser.add_argument('--context_size', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=20)
    parser.add_argument('--pool_size', type=int, default=-1)
    
    parser.add_argument('--save_label', default='eval')
    parser.add_argument('--save_path', default='./results')
    parser.add_argument('--ppd_path', default='oracle/resources/2023-02-07-ppd-mp.pkl.gz')
    
    parser.add_argument('--opt_goal', choices=['e_hull_distance', 'bulk_modulus_relaxed', 'multi-obj'], default='e_hull_distance')
    parser.add_argument('--task', choices=['csg', 'csp', 'csp_MnO2'], default='csg')
    parser.add_argument('--csp_compound', choices=['Ag6O2', 'Bi2F8', 'Co2Sb2', 'Co4B2'], default='Ag6O2')
    
    return parser.parse_args()


def initialize_models(args: argparse.Namespace) -> Tuple:
    """Initialize all required models."""
    
    base_path = Path(f'{args.save_path}/{args.save_label}')
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM
    llm_manager = LLMManager(
        args.base_model,
        args.tensor_parallel_size,
        args.gpu_memory_utilization,
        args.temperature,
        args.max_tokens
    )
    
    generator = StructureGenerator(llm_manager, base_path, args)
    
    evaluator = StructureEvaluator(base_path=base_path)
    mlip = "orb-v3" if args.task == "csp" else "chgnet" # "sevenet" "orb-v3" "chgnet" 
    stability_calculator = StabilityCalculator(mlip=mlip, ppd_path=args.ppd_path)
    
    # evaluator, stability_calculator = None, None
    
    return llm_manager, generator, evaluator, stability_calculator

def filter_valid_values(df, columns):
    """Create a single mask for multiple columns"""
    mask = True
    for col in columns:
        mask &= ~(df[col].isna() | df[col].isin([np.inf, -np.inf]) | (df[col] == 0))
    return mask

def matches_composition(comp, target_elements):
    """Check if composition contains all target elements (Ag, O) regardless of ratio."""
    # if not all(el in comp.elements for el in target_elements):
    #     return False
    # reduced_comp, _ = comp.get_reduced_composition_and_factor()
    # return all(abs(reduced_comp[el] - amt) <= 1e-6 for el, amt in target_ratio.items())
    return all(el in comp.elements for el in target_elements)

def initialize_task_data(evaluator: StructureEvaluator, args: argparse.Namespace):
    """Initialize and prepare task data."""

    seed_structures_df = pd.read_csv('oracle/resources/band_gap_processed.csv')
    seed_structures_df['structure'] = seed_structures_df['structure'].apply(lambda x: Structure.from_str(x, fmt='json') if pd.notna(x) else None)
    seed_structures_df['composition'] = [s.composition for s in seed_structures_df['structure']]
    seed_structures_df['composition_str'] = [s.composition.formula for s in seed_structures_df['structure']]
        
    # required_columns = ['structure', 'composition', 'composition_str', 'composition_len', 'e_hull_distance', 'delta_e', 'bulk_modulus', 'bulk_modulus_relaxed']
    required_columns = ['structure', 'composition', 'composition_str', 'composition_len', 'e_hull_distance', 'delta_e']
    # seed_structures_df = (seed_structures_df
    #     [(seed_structures_df['is_balanced'] == 1) & 
    #      (seed_structures_df['is_bond_valid'] == True) &
    #      (seed_structures_df['composition_len'].between(3, 6))]
    # )
    # Conditional Filtering
    if args.task == "csg":
        seed_structures_df = seed_structures_df.sort_values('e_hull_distance', ascending=True)
        seed_structures_df = seed_structures_df[np.isfinite(seed_structures_df['e_hull_distance'])]
        seed_structures_df = seed_structures_df[:args.pool_size]
        # seed_structures_df = seed_structures_df.drop_duplicates(subset='composition_str')
        # assert len(seed_structures_df) == args.pool_size
        # seed_structures_df = seed_structures_df[~seed_structures_df['composition_str'].str.contains('|'.join(LANTHANIDES + ACTINIDES), na=False)]
        # seed_structures_df = seed_structures_df[np.isfinite(seed_structures_df['e_hull_distance'])]
    elif args.task == "csp":
        target_comp = Composition(args.csp_compound)
        target_elements = set(target_comp.elements)
        target_ratio = {str(el): amt/target_comp.get_reduced_composition_and_factor()[1] 
                        for el, amt in target_comp.items()}
        # seed_structures_df = seed_structures_df[seed_structures_df['composition'].apply(matches_composition)]
        seed_structures_df = seed_structures_df[seed_structures_df['composition'].apply(lambda comp: matches_composition(comp, target_elements))]

    # elif args.task == "csp_MnO2":
    #     seed_structures_df = seed_structures_df[np.isfinite(seed_structures_df['e_hull_distance'])]
    #     seed_structures_df = seed_structures_df[seed_structures_df['composition'].apply(has_MnO2)]
    else:
        raise ValueError(f"Invalid task: {args.task}")    
    seed_structures_df = seed_structures_df[required_columns].sample(frac=1, random_state=args.random_seed)
    seed_structures_df['source'] = 'matbench'
    seed_structures_df['bulk_modulus'] = float(0.0)
    seed_structures_df['bulk_modulus_relaxed'] = float(0.0)
    print(f"Using extra pool of {len(seed_structures_df)} structures...")
    
    return seed_structures_df

def run_generation_iteration(
    iteration: int,
    curr_generation: GenerationResult,
    generator: StructureGenerator,
    evaluator: StructureEvaluator,
    stability_calculator: StabilityCalculator,
    args: argparse.Namespace
) -> GenerationResult:
    """Run a single generation iteration and return combined results."""
    print(f'Iteration {iteration}...')
    iter_start_time = time.time()

    llm_structures, llm_parents = generator.generate_structures(curr_generation.structure)
    num_a = len(llm_structures)
    print(f'Generated {num_a} structures')
    
    iter_eval_start_time = time.time()
    iter_gen_time = iter_eval_start_time - iter_start_time
    
    llm_crystals, llm_structures = evaluator.to_crys(llm_structures)
    num_b = len(llm_crystals)
    validity_b = evaluator.check_validity(llm_crystals)
    print(f'Converted {num_b} structures to crystals')
    
    
    # Filter both structures and parents
    llm_structures, llm_parents = evaluator.filter_balanced_structures(llm_structures, llm_parents)
    llm_crystals, llm_structures = evaluator.to_crys(llm_structures) # call to_crystals again
    num_c = len(llm_structures)
    print(f'Filtered to {num_c} balanced structures')
    
    # Get predictions and calculate objectives
    # llm_predictions = oracle.predict_by_structures(llm_structures) if len(llm_structures) else []
    
    # Compute stability
    stability_results = stability_calculator.compute_stability(llm_structures, wo_bulk=args.opt_goal=='e_hull_distance')
    if not stability_results:
        stability_results = [None] * len(llm_structures)
    
    generation_result = process_stability_results(stability_results, llm_structures)
    list_attrs = ['parents', 'composition', 'objective', 'e_hull_distance', 
                     'delta_e', 'crystal', 'bulk_modulus', 'structure_relaxed', 'bulk_modulus_relaxed']
    for attr in list_attrs:
        value = getattr(generation_result, attr)
        if value is not None:
            if len(value) != len(generation_result.structure):
                print(f"Warning: {attr} length mismatch. Expected {len(generation_result.structure)}, got {len(value)}")

    
    iter_end_time = time.time()
    iter_eval_time = iter_end_time - iter_eval_start_time
    iter_time = iter_end_time - iter_start_time
    
    print(f'Iteration {iteration} evaluation finished in {iter_eval_time:.2f} seconds')
    print(f'Completed iteration {iteration} in {iter_time:.2f} seconds')

    timing_data = {
        'generation_time': iter_gen_time,
        'evaluation_time': iter_eval_time,
        'iteration_time': iter_time
    }
    return GenerationResult(
        structure=generation_result.structure,
        parents=llm_parents,  
        composition=generation_result.composition,
        objective=generation_result.objective,
        e_hull_distance=generation_result.e_hull_distance,
        delta_e=generation_result.delta_e,
        source=['llm'] * len(generation_result.structure),
        crystal=llm_crystals,
        bulk_modulus=generation_result.bulk_modulus,
        structure_relaxed=generation_result.structure_relaxed,
        bulk_modulus_relaxed=generation_result.bulk_modulus_relaxed,
        validity_b=validity_b,
        num_a=num_a,
        num_b=num_b,
        num_c=num_c,
        timing_data=timing_data
    )


def normalized_weighted_sum(df, e_weight=0.7, b_weight=0.3, winsorize=True):
    e_values = df['e_hull_distance'].values
    b_values = df['bulk_modulus'].replace([float('-inf')], np.nan).fillna(0).values
    if winsorize:
        valid_e = e_values[~np.isnan(e_values)]
        e_upper = np.percentile(valid_e, 95) if len(valid_e) > 0 else np.max(valid_e)
        e_values = np.array([min(e, e_upper) if not np.isnan(e) else e for e in e_values])
    # Min-max normalization for e_hull_distance (minimize)
    valid_e = e_values[~np.isnan(e_values)]
    if len(valid_e) > 0:
        e_min = np.min(valid_e)
        e_max = np.max(valid_e)
        if e_max == e_min:
            e_normalized = np.zeros_like(e_values)
        else:
            e_normalized = np.array([(e - e_min) / (e_max - e_min) 
                                     if not np.isnan(e) else 1.0 for e in e_values])
    else:
        e_normalized = np.zeros_like(e_values)
    valid_b = b_values[~np.isnan(b_values) & (b_values > 0)]  # Ignore zeros from failed calculations
    if len(valid_b) > 0:
        b_min = np.min(valid_b)
        b_max = np.max(valid_b)
        if b_max == b_min:
            b_normalized = np.array([1.0 if (not np.isnan(b) and b > 0) else 0.0 for b in b_values])
        else:
            b_normalized = np.array([(b - b_min) / (b_max - b_min) 
                                    if (not np.isnan(b) and b > 0) else 0.0 for b in b_values])
    else:
        b_normalized = np.zeros_like(b_values)
    b_normalized_inverted = 1 - b_normalized
    objective = e_weight * e_normalized + b_weight * b_normalized_inverted
    return objective.tolist()

def lexicographic_ordering(df, e_threshold=0.03, b_scale=0.001, e_penalty=1.0):
    e_values = df['e_hull_distance'].values
    b_values = df['bulk_modulus'].replace([float('-inf')], np.nan).fillna(0).values
    n = len(e_values)
    objective = np.zeros(n)
    for i in range(n):
        e = e_values[i]
        b = b_values[i]
        if np.isnan(e):
            objective[i] = float('inf')
        else:
            if e <= e_threshold:
                # For structures with good e_hull_distance, prioritize bulk_modulus
                primary = e * 0.01 
                secondary = -b_scale * b if (not np.isnan(b) and b > 0) else 0
                objective[i] = primary + secondary
            else:
                objective[i] = e + e_penalty
    return objective.tolist()

def multi_objective_optimizer(df, method='lexical', **kwargs):
    if method.lower() == 'weighted':
        return normalized_weighted_sum(df, **kwargs)
    elif method.lower() == 'lexical':
        return lexicographic_ordering(df, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'weighted' or 'lexical'.")
        
    
def process_stability_results(stability_results, structures):
    result_dicts = []
    for r in stability_results:
        if r is not None:
            result_dict = {
                'e_hull_distance': r.e_hull_distance if r.e_hull_distance is not None else float('inf'),
                'delta_e': r.delta_e if r.delta_e is not None else float('inf'),
                'bulk_modulus': r.bulk_modulus if r.bulk_modulus is not None else float('-inf'),
                'bulk_modulus_relaxed': r.bulk_modulus_relaxed if r.bulk_modulus_relaxed is not None else float('-inf'),
                'structure_relaxed': getattr(r, 'structure_relaxed', None)
            }
        else:
            result_dict = {
                'e_hull_distance': float('inf'),
                'delta_e': float('inf'),
                'bulk_modulus': float(0.0),
                'bulk_modulus_relaxed': float(0.0),
                'structure_relaxed': None
            }
        result_dicts.append(result_dict)
    if len(result_dicts) == 0:
        return GenerationResult(**{field: [None] for field in GenerationResult.__dataclass_fields__})
    df = pd.DataFrame(result_dicts)
    return GenerationResult(
        structure=structures,
        composition=[s.composition for s in structures],
        objective=multi_objective_optimizer(df),
        e_hull_distance=df['e_hull_distance'].tolist(),
        delta_e=df['delta_e'].tolist(),
        bulk_modulus=df['bulk_modulus'].tolist(),
        structure_relaxed=df['structure_relaxed'].tolist(),
        bulk_modulus_relaxed=df['bulk_modulus_relaxed'].tolist()
    )

def get_parent_generation(evaluator, stability_calculator, input_generation: GenerationResult, parent_generation: GenerationResult,
                     full_df: pd.DataFrame, sort_target: str, args: argparse.Namespace, iter: int) -> GenerationResult:
    """Get sorted generation combining input structures and parent generation results."""
    interested_columns = ['structure', 'composition', 'composition_str', 'e_hull_distance', 'delta_e', 'bulk_modulus', 'bulk_modulus_relaxed', 'source']
    if input_generation and parent_generation: 
        generation_df = pd.DataFrame({
            'structure': input_generation.structure + parent_generation.structure,
            'composition': [s.composition for s in (input_generation.structure + parent_generation.structure)],
            'e_hull_distance': input_generation.e_hull_distance + parent_generation.e_hull_distance,
            'delta_e': input_generation.delta_e + parent_generation.delta_e,
            'bulk_modulus': input_generation.bulk_modulus + parent_generation.bulk_modulus,
            'bulk_modulus_relaxed': input_generation.bulk_modulus_relaxed + parent_generation.bulk_modulus_relaxed,
            'source': input_generation.source + parent_generation.source
        })
        ascending = (sort_target not in ['bulk_modulus', 'bulk_modulus_relaxed'])  
        sampled_seeds_df = full_df[interested_columns].sample(n=args.topk * args.context_size)
        # generation_df = pd.concat([generation_df, full_df[interested_columns]], ignore_index=True)
        generation_df = pd.concat([generation_df, sampled_seeds_df], ignore_index=True)

        generation_df['objective'] = multi_objective_optimizer(generation_df)
        generation_df = generation_df.sort_values(sort_target, ascending=ascending)
        generation_df['composition_str'] = generation_df['composition'].apply(lambda x: x.formula)
    else:
        generation_df = full_df[interested_columns].copy()
        generation_df['objective'] = multi_objective_optimizer(generation_df)

        generation_df = generation_df.sample(frac=1, random_state=args.random_seed)
    
    if args.task == "csg":
        generation_df = generation_df.drop_duplicates(subset='composition_str')
        # generation_df = generation_df[~generation_df['composition_str'].str.contains('|'.join(LANTHANIDES + ACTINIDES), na=False)]
    elif args.task == "csp":
        target_comp = Composition(args.csp_compound)
        target_elements = set(target_comp.elements)
        generation_df = generation_df[generation_df['composition'].apply(lambda comp: matches_composition(comp, target_elements))]

    else:
        raise ValueError(f"Invalid task: {args.task}")

    generation_df = generation_df.drop(columns=['composition_str'])

    print(f'Preparing {args.topk * args.context_size} parent structures for next generation from {len(generation_df)} structures...')
    generation_df = generation_df.iloc[:args.topk * args.context_size]
    indices = np.random.choice(len(generation_df), size=args.topk * args.context_size, replace=(len(generation_df) < args.topk * args.context_size))
    sampled_df = generation_df.iloc[indices]

    parents_df = sampled_df.copy()
    parents_file = Path(f'{args.save_path}/{args.save_label}/parents.csv')
    parents_df['structure'] = parents_df['structure'].apply(lambda x: Structure.to(x, fmt='json'))
    parents_df['iteration'] = iter
    parents_df.to_csv(parents_file, mode='a', header=not parents_file.exists(), index=False)
        
    return GenerationResult(
        structure=sampled_df['structure'].tolist(),
        bulk_modulus=sampled_df['bulk_modulus'].tolist(),
        bulk_modulus_relaxed=sampled_df['bulk_modulus_relaxed'].tolist(),
        e_hull_distance=sampled_df['e_hull_distance'].tolist(),
        delta_e=sampled_df['delta_e'].tolist(),
        source=sampled_df['source'].tolist(),
        objective=sampled_df['objective'].tolist()
    )
    # if input_generation and parent_generation:  # iterations after 1
    #     return GenerationResult(
    #         structure=sampled_df['structure'].tolist(),
    #         bulk_modulus=sampled_df['bulk_modulus'].tolist(),
    #         bulk_modulus_relaxed=sampled_df['bulk_modulus_relaxed'].tolist(),
    #         e_hull_distance=sampled_df['e_hull_distance'].tolist(),
    #         delta_e=sampled_df['delta_e'].tolist()
    #     )
    # else:  # first generation
    #     structures = sampled_df['structure'].tolist()
    #     # Compute stability
    #     stability_results = stability_calculator.compute_stability(structures)
    #     return process_stability_results(stability_results, structures)
        
def main():
    """Main execution function."""
    # Parse arguments and setup environment
    torch.cuda.empty_cache()
    args = parse_arguments()
    random.seed(args.random_seed)
    # Initialize models and components
    llm_manager, generator, evaluator, stability_calculator = initialize_models(args)

    
    # Initialize task data
    seed_structures_df = initialize_task_data(evaluator, args)
    # Select initial generation
    curr_generation = get_parent_generation(evaluator, stability_calculator, None, None, seed_structures_df, None, args, 0)
    
    # Initialize results tracking
    population_df = pd.DataFrame()
    all_generated_df = pd.DataFrame()
    eval_df = pd.DataFrame()
    start_time = time.time()
    
    for iteration in range(1, args.max_iter + 1):
        generation_result = run_generation_iteration(
            iteration,
            curr_generation,
            generator,
            evaluator,
            stability_calculator,
            args
        )
        
        # Calculate metrics for evaluation
        metrics = evaluator.evaluate_generation(generation_result, iteration=iteration, args=args)
        
        # Save results
        evaluator.save_results(generation_result, metrics, iteration=iteration, args=args)
        
        # Update current generation
        opt_goal = args.opt_goal
        if opt_goal == "multi-obj":
            # opt_goal = "e_hull_distance" if iteration % 2 == 1 else "bulk_modulus_relaxed"
            opt_goal = "objective"
        curr_generation = get_parent_generation(evaluator, stability_calculator, generation_result, curr_generation, seed_structures_df, opt_goal, args, iteration)
        # e_hull_distance or bulk_modulus_relaxed
        print(f'Completed iteration {iteration}')
        print(f'Current metrics: {metrics}')
        
    
    total_time = time.time() - start_time
    print(f'Completed all iterations in {total_time:.2f} seconds')

if __name__ == "__main__":
    main()






    