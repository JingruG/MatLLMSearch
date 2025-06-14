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
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

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
    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--reproduction_size', type=int, default=5)
    parser.add_argument('--context_size', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=20)
    parser.add_argument('--pool_size', type=int, default=-1)
    
    parser.add_argument('--save_label', default='eval')
    parser.add_argument('--save_path', default='./results')
    parser.add_argument('--ppd_path', default='oracle/resources/2023-02-07-ppd-mp.pkl.gz')
    
    parser.add_argument('--opt_goal', choices=['e_hull_distance', 'bulk_modulus_relaxed', 'multi-obj'], default='e_hull_distance')
    parser.add_argument('--task', choices=['csg', 'csp', 'csg_zeroshot'], default='csg')
    parser.add_argument('--csp_compound', choices=['Ag6O2', 'Bi2F8', 'Co2Sb2', 'Co4B2', 'Cr4Si4', 'KZnF3', "Sr2O4", "YMg3"], default='Ag6O2')
    
    parser.add_argument('--resume', type=str, default='')
    
    return parser.parse_args()


def initialize_models(args: argparse.Namespace) -> Tuple:
    """Initialize all required models."""
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda:0')
    else:
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    
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
    # mlip = "chgnet" # "sevenet" "orb-v3" "chgnet" 
    stability_calculator = StabilityCalculator(mlip=mlip, ppd_path=args.ppd_path, device=device)
    
    # evaluator, stability_calculator = None, None
    
    return llm_manager, generator, evaluator, stability_calculator

def filter_valid_values(df, columns):
    """Create a single mask for multiple columns"""
    mask = True
    for col in columns:
        mask &= ~(df[col].isna() | df[col].isin([np.inf, -np.inf]) | (df[col] == 0))
    return mask
    
def contains_elements(comp, target_comp):
    """Check if composition contains all target elements regardless of ratio."""
    return all(el in comp.elements for el in target_comp.elements)
    
def matches_composition(comp, target_comp):
    """Check if composition has exactly the same elements and atom counts as target."""
    if set(comp.elements) != set(target_comp.elements):
        return False
    return all(abs(comp[el] - target_comp[el]) <= 1e-6 for el in target_comp.elements)
    
def matches_unit_cell_pattern(comp1, comp2):
    """Check if two compositions have the same number of elements and exact atom count pattern."""
    if len(comp1.elements) != len(comp2.elements):
        return False
    total_atoms1 = sum(comp1.values())
    total_atoms2 = sum(comp2.values())
    if total_atoms1 != total_atoms2:
        return False
    counts1 = sorted([comp1[el] for el in comp1.elements])
    counts2 = sorted([comp2[el] for el in comp2.elements])
    return counts1 == counts2
    
def initialize_task_data(evaluator: StructureEvaluator, args: argparse.Namespace):
    """Initialize and prepare task data."""

    if "zeroshot" in args.task:
        seed_structures_df = pd.read_csv('results/zero-one/poscar_70b_1000_zeroshot/generations.csv')
        seed_structures_df['e_hull_distance'] = seed_structures_df['EHullDistance']
        seed_structures_df['structure'] = seed_structures_df['StructureRelaxed']
        seed_structures_df['source'] = 'llm'
        seed_structures_df['delta_e'] = seed_structures_df['DeltaE'].fillna(float('inf'))
    else:
        seed_structures_df = pd.read_csv('oracle/resources/band_gap_processed.csv')
        seed_structures_df['source'] = 'matbench'
        seed_structures_df['delta_e'] = float('inf')
    seed_structures_df['structure'] = seed_structures_df['structure'].apply(lambda x: Structure.from_str(x, fmt='json') if pd.notna(x) else None)
    seed_structures_df['composition'] = [s.composition for s in seed_structures_df['structure']]
    seed_structures_df['composition_str'] = [s.composition.formula for s in seed_structures_df['structure']]
    seed_structures_df['composition_len'] = [len(s.composition.elements) for s in seed_structures_df['structure']]
    seed_structures_df['bulk_modulus'] = float(0.0)
    seed_structures_df['bulk_modulus_relaxed'] = float(0.0)
    required_columns = ['structure', 'composition', 'composition_str', 'composition_len', 'e_hull_distance', 'delta_e', 'source', 'bulk_modulus', 'bulk_modulus_relaxed']
    seed_structures_df = seed_structures_df[required_columns].sample(frac=1, random_state=args.random_seed)
    seed_structures_df = seed_structures_df[seed_structures_df['composition_len'].between(3, 6)]
    if "csg" in args.task:
        seed_structures_df = seed_structures_df.sort_values('e_hull_distance', ascending=True)
        seed_structures_df = seed_structures_df[np.isfinite(seed_structures_df['e_hull_distance'])]
        seed_structures_df = seed_structures_df[:args.pool_size]
        # seed_structures_df = seed_structures_df.drop_duplicates(subset='composition_str')
        # assert len(seed_structures_df) == args.pool_size
        # seed_structures_df = seed_structures_df[~seed_structures_df['composition_str'].str.contains('|'.join(LANTHANIDES + ACTINIDES), na=False)]
        # seed_structures_df = seed_structures_df[np.isfinite(seed_structures_df['e_hull_distance'])]
    elif args.task == "csp":
        target_comp = Composition(args.csp_compound)
        # seed_structures_df = seed_structures_df[seed_structures_df['composition'].apply(matches_composition)]
        seed_structures_df = seed_structures_df[seed_structures_df['composition'].apply(lambda comp: matches_unit_cell_pattern(comp, target_comp))]
    else:
        raise ValueError(f"Invalid task: {args.task}")    
    print(f"A total of {len(seed_structures_df)} structures as reference...")
    
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
    llm_structures, llm_parents = evaluator.filter_balanced_structures(llm_structures, llm_parents, args.task)
    if args.task == "csp":
        target_comp = Composition(args.csp_compound)
        filtered_data = [(s, p) for s, p in zip(llm_structures, llm_parents) if matches_composition(s.composition, target_comp)]
        
        llm_structures, llm_parents = zip(*filtered_data) if filtered_data else ([], [])
        llm_structures, llm_parents = list(llm_structures), list(llm_parents)
    llm_crystals, llm_structures = evaluator.to_crys(llm_structures) # call to_crystals again
    num_c = len(llm_structures)
    print(f'Filtered to {num_c} balanced structures')

    # Compute stability
    stability_results = stability_calculator.compute_stability(llm_structures, wo_bulk=args.opt_goal=='e_hull_distance')
    if not stability_results:
        stability_results = [None] * len(llm_structures)
    
    generation_result = process_stability_results(stability_results, llm_structures)
    list_attrs = ['parents', 'composition', 'objective', 'e_hull_distance', 'energy','energy_relaxed', 'delta_e', 'crystal', 'bulk_modulus', 'structure_relaxed', 'bulk_modulus_relaxed']
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
        energy=generation_result.energy,
        energy_relaxed=generation_result.energy_relaxed,
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
            e_normalized = np.array([(e - e_min) / (e_max - e_min) if not np.isnan(e) else 1.0 for e in e_values])
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
                'energy': r.energy if r.energy is not None else float('inf'),
                'energy_relaxed': r.energy_relaxed if r.energy_relaxed is not None else float('inf'),
                'bulk_modulus': r.bulk_modulus if r.bulk_modulus is not None else float('-inf'),
                'bulk_modulus_relaxed': r.bulk_modulus_relaxed if r.bulk_modulus_relaxed is not None else float('-inf'),
                'structure_relaxed': getattr(r, 'structure_relaxed', None)
            }
        else:
            result_dict = {
                'e_hull_distance': float('inf'),
                'delta_e': float('inf'),
                'energy': float('inf'),
                'energy_relaxed': float('inf'),
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
        objective=multi_objective_optimizer(df, method='weighted'),
        energy=df['energy'].tolist(),
        energy_relaxed=df['energy_relaxed'].tolist(),
        e_hull_distance=df['e_hull_distance'].tolist(),
        delta_e=df['delta_e'].tolist(),
        bulk_modulus=df['bulk_modulus'].tolist(),
        structure_relaxed=df['structure_relaxed'].tolist(),
        bulk_modulus_relaxed=df['bulk_modulus_relaxed'].tolist()
    )

def get_parent_generation(evaluator, stability_calculator, input_generation: GenerationResult, parent_generation: GenerationResult,
                     full_df: pd.DataFrame, sort_target: str, args: argparse.Namespace, iter: int) -> GenerationResult:
    """Get sorted generation combining input structures and parent generation results."""
    if input_generation is None and parent_generation is None and full_df is None: # zero-shot only
        return GenerationResult(**{field: [None] for field in GenerationResult.__dataclass_fields__})
    interested_columns = ['structure', 'composition', 'composition_str', 'e_hull_distance', 'delta_e', 'bulk_modulus', 'bulk_modulus_relaxed', 'source']
    generation_df = pd.DataFrame(columns=interested_columns)
    # Combine input and parent generations when available
    if input_generation and len(input_generation.structure) > 0 and input_generation.structure[0] is not None:
        input_source = input_generation.source if hasattr(input_generation, 'source') and input_generation.source else ['llm'] * len(input_generation.structure)
        input_df = pd.DataFrame({
            'structure': input_generation.structure,
            'composition': [s.composition for s in input_generation.structure],
            'e_hull_distance': input_generation.e_hull_distance if hasattr(input_generation, 'e_hull_distance') else [float('inf')] * len(input_generation.structure),
            'delta_e': input_generation.delta_e if hasattr(input_generation, 'delta_e') else [float('inf')] * len(input_generation.structure),
            'bulk_modulus': input_generation.bulk_modulus if hasattr(input_generation, 'bulk_modulus') else [float('-inf')] * len(input_generation.structure),
            'bulk_modulus_relaxed': input_generation.bulk_modulus_relaxed if hasattr(input_generation, 'bulk_modulus_relaxed') else [float('-inf')] * len(input_generation.structure),
            'source': input_source
        })
        generation_df = pd.concat([generation_df, input_df], ignore_index=True)
    if parent_generation and len(parent_generation.structure) > 0 and parent_generation.structure[0] is not None:
        parent_source = parent_generation.source if hasattr(parent_generation, 'source') and parent_generation.source else ['parent'] * len(parent_generation.structure)
        parent_df = pd.DataFrame({
            'structure': parent_generation.structure,
            'composition': [s.composition for s in parent_generation.structure],
            'e_hull_distance': parent_generation.e_hull_distance if hasattr(parent_generation, 'e_hull_distance') else [float('inf')] * len(parent_generation.structure),
            'delta_e': parent_generation.delta_e if hasattr(parent_generation, 'delta_e') else [float('inf')] * len(parent_generation.structure),
            'bulk_modulus': parent_generation.bulk_modulus if hasattr(parent_generation, 'bulk_modulus') else [float('-inf')] * len(parent_generation.structure),
            'bulk_modulus_relaxed': parent_generation.bulk_modulus_relaxed if hasattr(parent_generation, 'bulk_modulus_relaxed') else [float('-inf')] * len(parent_generation.structure),
            'source': parent_source
        })
        generation_df = pd.concat([generation_df, parent_df], ignore_index=True)
        
    generation_df = generation_df[generation_df['e_hull_distance'] <= 0.1] # only keep metastable ones from past generations
    needed_count = args.population_size * args.context_size - len(generation_df)
    if needed_count > 0 and full_df is not None and len(full_df) > 0:
        sample_size = min(needed_count, len(full_df))
    #     sampled_seeds_df = full_df[interested_columns].sample(n=sample_size)
    #     generation_df = pd.concat([generation_df, sampled_seeds_df], ignore_index=True)
        generation_df = pd.concat([generation_df, full_df[interested_columns][:needed_count]], ignore_index=True)
    
    generation_df['objective'] = multi_objective_optimizer(generation_df)
    generation_df['composition_str'] = generation_df['composition'].apply(lambda x: x.formula)
    ascending = (sort_target not in ['bulk_modulus', 'bulk_modulus_relaxed'])        
    generation_df = generation_df.sort_values(sort_target, ascending=ascending)
    
    if "csg" in args.task:
        generation_df = generation_df.drop_duplicates(subset='composition_str')
        # generation_df = generation_df[~generation_df['composition_str'].str.contains('|'.join(LANTHANIDES + ACTINIDES), na=False)]
    elif args.task == "csp":
        target_comp = Composition(args.csp_compound)
        target_elements = set(target_comp.elements)
        generation_df = generation_df[generation_df['composition'].apply(lambda comp: matches_unit_cell_pattern(comp, target_comp))]

    else:
        raise ValueError(f"Invalid task: {args.task}")

    generation_df = generation_df.drop(columns=['composition_str'])

    print(f'Preparing {args.population_size * args.context_size} parent structures for next generation from {len(generation_df)} structures...')
    generation_df = generation_df.iloc[:args.population_size * args.context_size]
    indices = np.random.choice(len(generation_df), size=args.population_size * args.context_size, replace=(len(generation_df) < args.population_size * args.context_size))
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
    
def resume_from_checkpoint(args):
    """Load previous generations and resume from the last completed iteration."""
    # Load generation history
    generations_path = Path(f'{args.save_path}/{args.resume}/generations.csv')
    generations_df = pd.read_csv(generations_path)
    last_iter = generations_df['Iteration'].max()
    print(f"Resuming iteration {last_iter} from checkpoint: {args.resume}")
    last_iter_generations = generations_df[generations_df['Iteration'] == last_iter]
    structures = []
    for _, row in last_iter_generations.iterrows():
        try:
            structure = Structure.from_str(row['StructureRelaxed'], fmt='json')
            structures.append(structure)
        except Exception as e:
            print(f"Error loading structure: {e}")
    
    # Create generation result object
    generation_result = GenerationResult(
        structure=structures,
        parents=[None] * len(structures),  # No parent information for seeds
        composition=[s.composition for s in structures],
        e_hull_distance=last_iter_generations['EHullDistance'].tolist() if 'EHullDistance' in last_iter_generations.columns else [0.0] * len(structures),
        delta_e=[0.0] * len(structures),
        energy=[0.0] * len(structures),
        energy_relaxed=[0.0] * len(structures),
        source='llm',
        bulk_modulus=[0.0] * len(structures),
        bulk_modulus_relaxed=[0.0] * len(structures),
        structure_relaxed=structures, 
        objective=[0.0] * len(structures),
    )
    
    return generation_result, last_iter   
        
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
    start_iter = 1
    if args.resume:
        curr_generation, last_iter = resume_from_checkpoint(args)
        curr_generation = get_parent_generation(evaluator, stability_calculator, None, curr_generation, seed_structures_df, args.opt_goal, args, last_iter)
        start_iter = last_iter + 1
    else:
        curr_generation = get_parent_generation(evaluator, stability_calculator, None, None, seed_structures_df, args.opt_goal, args, 0)

    # Initialize results tracking
    start_time = time.time()
    for iteration in range(start_iter, args.max_iter + 1):
        generation_result = run_generation_iteration(
            iteration,
            curr_generation,
            generator,
            evaluator,
            stability_calculator,
            args
        )
        
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






    