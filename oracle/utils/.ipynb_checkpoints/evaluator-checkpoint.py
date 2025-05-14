import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from pathlib import Path
from bs4 import BeautifulSoup
import re
import json
from utils.config import METAL_OXIDATION_STATES
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core.structure import Structure
from utils.generator import GenerationResult
from dataclasses import dataclass
from utils.basic_eval import structure_to_crystal, get_fp_pdist, CompScaler, timeout, TimeoutError
from collections import Counter


class StructureEvaluator:
    def __init__(self, base_path: Path):
        self.base_path = base_path

    @staticmethod
    def to_crys(structures: List[Structure]) -> Tuple[List, List]:
        crys, strucs = [], []
        for s in structures:
            try:
                if cry := structure_to_crystal(s):
                    crys.append(cry)
                    strucs.append(s)
            except Exception as e:
                print(f'Crystal conversion error: {e}')
        return crys, strucs

    @staticmethod
    def check_validity(crys: List) -> Dict[str, float]:
        if not crys:
            return {'comp_valid': 0.0, 'struct_valid': 0.0, 'valid': 0.0}
        return {
            'comp_valid': np.mean([c.comp_valid for c in crys]),
            'struct_valid': np.mean([c.struct_valid for c in crys]),
            'valid': np.mean([c.valid for c in crys])
        }

    @staticmethod
    def check_diversity(structures: List) -> Dict[str, float]:
        # valid_crys = [c for c in crys if c.valid]
        # metrics = {'comp_div': 0.0, 'struct_div': 0.0}
        # if valid_crys:
        #     comp_fps = [c.comp_fp for c in valid_crys]
        #     max_len = max(len(fp) for fp in comp_fps)
        #     comp_fps = [np.pad(fp, (0, max_len - len(fp)), 'constant') for fp in comp_fps]
        #     comp_fps = CompScaler.transform(comp_fps)
        #     metrics.update({'comp_div': get_fp_pdist(comp_fps)})
        #     struct_fps = get_fp_pdist([c.struct_fp for c in valid_crys])
        #     metrics.update({'struct_div': struct_fps})
        metrics = {'comp_div': 0.0}
        if not structures or len(structures) < 2:
            return metrics
        comp_strs = [str(s.composition.formula) for s in structures if s is not None]
        unique_comps = set(comp_strs)
        diversity = len(unique_comps) / len(structures)
        metrics.update({'comp_div': diversity})
        return metrics

    def valid_value(self, x) -> bool:
        """Check if value is valid (not None, not inf/-inf, not nan, not 0)."""
        return (x is not None and 
                not np.isinf(x) and 
                not np.isnan(x) and 
                x != 0)
    
    def valid_mean(self, values: List[float]) -> float:
        """Calculate mean of valid values, return 0.0 if no valid values."""
        valid_values = [x for x in values if self.valid_value(x)]
        return np.mean(valid_values) if valid_values else 0.0
    
    def check_stability(self, e_hull_distances: List[float]) -> Dict[str, Any]:
        """Calculate stability metrics for a generation."""
        metrics = {}
        
        # Calculate metrics for different thresholds
        for threshold in [0.00, 0.03, 0.1]:
            stabilities = [d < threshold for d in e_hull_distances if self.valid_value(d)]
            threshold_str = f"{threshold:.2f}"
            metrics.update({
                f"stable_rate_{threshold_str}": np.mean(stabilities) if stabilities else 0.0,
                f"stable_num_{threshold_str}": sum(stabilities),
            })
        return metrics
        
    def evaluate_generation(self, generation: GenerationResult, iteration: int, args: Any) -> Dict[str, Any]:
        """Evaluate generation metrics from GenerationResult."""
        metrics = {
            'iteration': iteration,
            'raw_valid': generation.validity_b['valid'],
            'raw_comp_valid': generation.validity_b['comp_valid'],
            'raw_struct_valid': generation.validity_b['struct_valid'],
            'num_a': generation.num_a,
            'num_b': generation.num_b,
            'num_c': generation.num_c,
            'num_structures': len(generation.structure),
            'num_valid_structures': len(generation.crystal),
            'avg_bulk_modulus': self.valid_mean(generation.bulk_modulus),
            'avg_bulk_modulus_relaxed': self.valid_mean(generation.bulk_modulus_relaxed),
            'avg_delta_e': self.valid_mean(generation.delta_e),
            'avg_ehull_dist': self.valid_mean(generation.e_hull_distance),
            'size_population': args.topk,
            'reproduction_size': args.reproduction_size
        }
        
        metrics.update(self.check_validity(generation.crystal))
        metrics.update(self.check_stability(generation.e_hull_distance))
        metrics.update(self.check_diversity(generation.structure))
        
        return metrics
        
    def no_isolate_atom(self, structure: Structure) -> bool:
        """Check if structure has no isolated atoms."""
        for i, site_i in enumerate(structure):
            bond_count = 0
            for j, site_j in enumerate(structure):
                if i != j:
                    dist = structure.get_distance(i, j)
                    r_A = CovalentRadius.radius[site_i.species_string]
                    r_B = CovalentRadius.radius[site_j.species_string]
                    if 0.6 * (r_A + r_B) > dist:
                        return False  # Too close
                    elif dist <= 1.3 * (r_A + r_B) and dist < 8:
                        bond_count += 1
            if bond_count < 1:
                return False
        return True

    
    # def validate_structure(self, structure: Structure) -> bool:
    #     """Validate structure based on physical and chemical constraints."""
    #     try:
    #         if not structure.composition.valid:
    #             print('Composition validity check fails')
    #             return False
    #         # Basic property checks
    #         if structure.composition.num_atoms <= 0:
    #             print('Atom number check fails')
    #             return False
                
    #         # Volume and density checks
    #         if structure.volume <= 0 or structure.volume >= 30 * structure.composition.num_atoms:
    #             print('Volume check fails')
    #             return False
                
    #         # Chemical checks
    #         # avg_electroneg = structure.composition.average_electroneg
    #         # if avg_electroneg > 3.5 or avg_electroneg < 1.0:
    #         #     return False
                
    #         # Structural checks
    #         positions = np.array([site.coords for site in structure])
    #         cog = positions.mean(axis=0)
    #         if not np.all(np.abs(cog - np.round(cog)) <= 1):
    #             print('Structural check fails')
    #             return False
                
    #         # Isolation check
    #         if not self.no_isolate_atom(structure):
    #             print('Isolation check fails')
    #             return False
                
    #         return True
    #     except Exception as e:
    #         print(f"Error validating structure: {e}")
    #         return False

    # def calculate_score(self, objective: float, e_hull_d: float, max_objective: float, max_e_hull_d: float, 
    #                    weights: dict = {'objective': 0.0, 'stability': 1.0}) -> float:
    #     """Calculate normalized weighted score."""
    #     # Normalize values
    #     def safe_normalize(val, max_val):
    #         return (1 - val/max_val) if val is not None and max_val != 0 else 0.0
        
    #     objective_score = safe_normalize(objective, max_objective)
    #     stability_score = safe_normalize(e_hull_d, max_e_hull_d)
        
    #     return (weights['objective'] * objective_score + 
    #             weights['stability'] * stability_score)
    
    # def sort_structures(self, generation: Any, weights: dict = None) -> dict:
    #     """Sort structures by weighted score."""
    #     fields = ['index', 'score', 'structure', 'composition', 'objective',
    #       'e_hull_distance', 'delta_e', 'bulk_modulus', 'structure_relaxed']
    #     valid_structs = [(idx, struct, obj, e_hull_d) 
    #                      for idx, (struct, obj, e_hull_d) in enumerate(zip(generation.structure, 
    #                                                                    generation.objective,
    #                                                                    generation.e_hull_distance))
    #                      if self.validate_structure(struct)]
        
    #     if not valid_structs:
    #         return {f: [] for f in fields}
        
    #     max_objective = max((x[2] for x in valid_structs if x[2] is not None), default=1.0)
    #     max_e_hull = max((x[3] for x in valid_structs if x[3] is not None), default=1.0)
    #     scored_data = [(
    #         idx,
    #         self.calculate_score(obj, e_hull, max_objective, max_e_hull, weights),
    #         struct,
    #         *[getattr(generation, attr, [None] * len(generation.structure))[idx]
    #           for attr in fields[3:]]
    #     ) for idx, struct, obj, e_hull in valid_structs]
        
    #     # Sort by score and objective
    #     sorted_data = sorted(scored_data, 
    #                         key=lambda x: (x[1], x[-2] if x[-2] is not None else 0.0),
    #                         reverse=True)
        
    #     return {f: [d[i] for d in sorted_data] for i, f in enumerate(fields)}
        
    def save_results(self, generation: Any, metrics: Dict[str, Any], iteration: int, args: Any) -> None:
        """Save generation results and metrics to CSV files."""
        print(len(generation.structure),len(generation.parents), ' structures saved to file ')   
        if len(generation.structure) and len(generation.structure) == len(generation.parents):
            generation_df = pd.DataFrame({
                'Iteration': [iteration] * len(generation.structure),
                'Structure': [json.dumps(s.as_dict(), sort_keys=True) if s is not None else None 
                             for s in generation.structure],
                'ParentStructures': [json.dumps([json.dumps(p.as_dict(), sort_keys=True) if p is not None else None for p in pp])
                             for pp in generation.parents],
                'Objective': generation.objective,
                'Composition': generation.composition,
                'DeltaE': generation.delta_e,
                'EHullDistance': generation.e_hull_distance,
                'BulkModulus': generation.bulk_modulus,
                'StructureRelaxed': [json.dumps(s.as_dict(), sort_keys=True) if s is not None else None 
                                    for s in generation.structure_relaxed],
                'BulkModulusRelaxed': generation.bulk_modulus_relaxed,
            })
            if hasattr(generation, 'timing_data') and generation.timing_data is not None:
                for key, value in generation.timing_data.items():
                    metrics[key] = value
            for path, data in [
                (f"generations.csv", generation_df),
                (f"metrics.csv", pd.DataFrame([metrics]))
            ]:
                full_path = self.base_path / path
                data.to_csv(full_path, mode='a', header=not full_path.exists(), index=False)
            
    @timeout(30, error_message="Balance composition check timed out after 30 seconds")
    def check_balanced_composition(self, structure: Structure):
        balanced_combinations = structure.composition.oxi_state_guesses(
            oxi_states_override=METAL_OXIDATION_STATES
        )
        if balanced_combinations:
            return True
        return False

    def filter_balanced_structures(self, structure_list: List[Structure], parents_list: List[List[Structure]]) -> Tuple[List[Structure], Optional[List[List[Structure]]]]:
        """Filter structures to keep only those with balanced charges, along with their parents."""
        balanced_structures, balanced_parents = [], []
        
        for idx, structure in enumerate(structure_list):
            try:
                structure = structure.get_primitive_structure()
                if not structure.is_3d_periodic:
                    print('is not 3d periodic')
                    continue
                # if self.no_isolate_atom(structure):
                #     print('has isolated atom')
                #     continue
                if self.check_balanced_composition(structure) and structure not in balanced_structures:
                    balanced_structures.append(structure)
                    balanced_parents.append(parents_list[idx])
            except Exception as e:
                print(f'Error checking charge balance: {e}')
                continue
        return balanced_structures, balanced_parents
