from utils.data_processing import truncate_text, correct_json_string
# from utils.structure_utils import filter_hypothesis, filter_balanced_structures
from utils.config import PROMPT_PATTERN_CSG, PROMPT_PATTERN_CSP, PROMPT_CRYSTALLLM
# from utils.evaluator import check_properties_input
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import time
from dataclasses import dataclass
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from bs4 import BeautifulSoup
import re
import os
import json
import random
from pathlib import Path
import numpy as np


def parse_fn(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]
    structure = Structure(
        lattice=Lattice.from_parameters(
            *(lengths + angles)),
        species=species,
        coords=coords, 
        coords_are_cartesian=False,
    )
    return structure.to(fmt="cif")
    
@dataclass
class GenerationResult:
    structure: List[Any]
    parents: List[Any] = None
    composition: Optional[List[Any]] = None
    objective: Optional[List[float]] = None
    e_hull_distance: Optional[List[float]] = None
    delta_e: Optional[List[float]] = None  
    source: Optional[Any] = None  
    crystal: Optional[List[Any]] = None
    bulk_modulus: Optional[List[float]] = None
    structure_relaxed: Optional[List[Structure]] = None
    bulk_modulus_relaxed: Optional[List[float]] = None
    validity_b: Optional[Any] = None
    num_a: Optional[Any] = None
    num_b: Optional[Any] = None
    num_c: Optional[Any] = None
    timing_data: Optional[Any] = None 
    

class StructureGenerator:
    def __init__(self, llm_manager: Any, base_path: Path, args: Any, fmt: str = 'poscar'):
        self.llm_manager = llm_manager
        self.args = args
        self.base_path = base_path
        self.fmt = args.fmt
        self.prompt_pattern = PROMPT_CRYSTALLLM if "crystalllm" in args.base_model else PROMPT_PATTERN_CSG if args.task == "csg" else PROMPT_PATTERN_CSP

        
    def _prepare_instructions(self, input_structure_strs: List) -> List:
        instructions = []
        for input_str in input_structure_strs:
            question = self.prompt_pattern.format(input=truncate_text(input_str, max_tokens=11800), rep_size=self.args.reproduction_size, fmt=self.fmt)
            question = truncate_text(question, max_tokens=11000)
            instructions.append(question)
        return instructions

    def structure_to_string(self, structure: Structure, precision: int = 12, fmt: str = 'poscar') -> str:
        """Convert Structure to formatted string with specified decimal precision"""
        fmt = fmt.lower()
        if fmt not in ['poscar', 'cif']:
            raise ValueError("Format must be either 'poscar' or 'cif'")
        if fmt == 'poscar' and precision < 12:
            species = []
            counts = []
            current_sp = None
            count = 0
            for site in structure:
                if site.species_string != current_sp:
                    if current_sp is not None:
                        species.append(current_sp)
                        counts.append(count)
                    current_sp = site.species_string
                    count = 1
                else:
                    count += 1
            species.append(current_sp)
            counts.append(count)
            fmt_str = f"{{:.{precision}f}}"
            lines = [
                " ".join(f"{sp}{cnt}" for sp, cnt in zip(species, counts)),  # Formula line
                "1.0",  # Scale factor
                # Lattice vectors with aligned spacing
                "\n".join("  " + " ".join(fmt_str.format(x) for x in row) 
                         for row in structure.lattice.matrix),
                " ".join(species),  # Species symbols
                " ".join(map(str, counts)),  # Counts
                "direct",  # Coordinate type
                # Site coordinates with aligned spacing and species
                "\n".join("   " + " ".join(fmt_str.format(x) for x in site.frac_coords) + 
                         f" {site.species_string}" for site in structure)
            ]
            return "\n".join(lines)
        
        return str(structure.to(fmt=fmt))

    def structures_to_json(self, structures: list[Structure], precision: int = 12, fmt: str = 'poscar') -> str:
        structures_dict = {}
        fmt = fmt.lower()
        if fmt not in ['poscar', 'cif']:
            raise ValueError("Format must be either 'poscar' or 'cif'")
        structures_dict = {
            str(i): {
                "formula": struct.composition.reduced_formula,
                # "formula": struct.composition.formula,
                fmt: self.structure_to_string(struct, precision=precision, fmt=fmt)
            }
            for i, struct in enumerate(structures)
        }
        return json.dumps(structures_dict, indent=2)
        
    def get_crystal_string(self, structure):
        # structure = Structure.from_str(cif_str, fmt="cif")
    
        # Randomly translate within the unit cell
        structure.translate_sites(
            indices=range(len(structure.sites)), vector=np.random.uniform(size=(3,))
        )
    
        lengths = structure.lattice.parameters[:3]
        angles = structure.lattice.parameters[3:]
        atom_ids = structure.species
        frac_coords = structure.frac_coords
    
        crystal_str = \
            " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
            " ".join([str(int(x)) for x in angles]) + "\n" + \
            "\n".join([
                str(t) + "\n" + " ".join([
                    "{0:.2f}".format(x) for x in c
                ]) for t,c in zip(atom_ids, frac_coords)
            ])
    
        return crystal_str
        
    def parse_fn(self, gen_str):
        lines = [x for x in gen_str.split("\n") if len(x) > 0]
        lengths = [float(x) for x in lines[0].split(" ")]
        angles = [float(x) for x in lines[1].split(" ")]
        species = [x for x in lines[2::2]]
        coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]
        
        structure = Structure(
            lattice=Lattice.from_parameters(
                *(lengths + angles)),
            species=species,
            coords=coords, 
            coords_are_cartesian=False,
        )
        return structure.to(fmt="cif")
        
    def generate_structures(self, input_structures: List) -> List:
        if "flowmm" in self.args.base_model:
            generated_structures = self.llm_manager.generate(input_structures)
            return list(map(lambda x: [x] for x in generated_structures)), list(map(lambda x: [x] for x in input_structures))
        input_groups = [input_structures[i:i + self.args.context_size] for i in range(0, len(input_structures), self.args.context_size)]
        
        input_structure_strs = [self.structures_to_json(input_group, precision=12, fmt=self.fmt) for input_group in input_groups]
        if "crystalllm" in self.args.base_model and self.fmt=="cif":
            input_structure_strs = []
            species_to_remove_list = []
            for input_group in input_groups:
                masked_structures = []
                for structure in input_group:
                    species = [str(s) for s in structure.species]
                    species_to_remove = random.choice(species)
                    species_to_remove_list.append(species_to_remove)
                    crystal_string = self.get_crystal_string(structure)  # Assuming this works on individual structures
                    partial_crystal_str = crystal_string.replace(
                        species_to_remove, "[MASK]"
                    )
                    masked_structures.append(partial_crystal_str)
                input_structure_strs.append(masked_structures)
        
        # if self.args.task == "csp_MnO2":
        #     host_struct = Structure.from_file('oracle/MnO2-host/mp-19395-alpha-MnO2-sym.cif')
        #     host_structure_str = self.structures_to_json([host_struct], precision=12, fmt=self.fmt)
        #     instructions = self._prepare_instructions(input_structure_strs, host_structure_str)
        # else:
        instructions = self._prepare_instructions(input_structure_strs)
        
        start = time.time()
        data_gathered = self.llm_manager.generate(instructions)
        print(f"Generation time elapsed: {time.time()-start}")
        print(len(data_gathered))
        return self.process_generated_data(data_gathered, input_groups)
        

    def remove_duplicate_input(self, structure_list: List[Structure], input_structures: List[Structure]) -> List[Structure]:
        unique_structures = []
        for structure in structure_list:
            try:
                # Check structure validity
                if not hasattr(structure, 'frac_coords') or structure.frac_coords.shape[1] != 3:
                    print(f"Error remove_duplicate_input: frac coordinates error")
                    continue
                    
                # Check if it matches any input structure
                if any(input_structure is not None and 
                      structure.matches(input_structure, scale=True, attempt_supercell=False) 
                      for input_structure in input_structures):
                    print(f"Error remove_duplicate_input: identical to input structure")
                    continue
                    
                # Check if it matches any already accepted structure
                if not any(structure.matches(s, scale=True, attempt_supercell=False) 
                          for s in unique_structures):
                    unique_structures.append(structure)
                    # print(f"Added new unique structure")
                    
            except Exception as e:
                print(f"Error remove_duplicate_input: {e}")
                return unique_structures
                continue
                
        return unique_structures
        
    def process_generated_data(self, data_gathered, input_structures_list):
        """The only change needed is in error handling since we're now dealing with JSON parsing errors"""
        # Process the JSON formatted data
        try:
            structures_gathered, error_res, hard_responses = map(list, zip(*[self.filter_hypothesis(data) for data in data_gathered]))
        except Exception as e:
            print(f"Error processing JSON data: {e}")
            return []       
        non_empty_batches = sum(1 for structs in structures_gathered if len(structs) > 0)
        print(f"Batches with structures: {non_empty_batches}/{len(structures_gathered)}")

        self.save_error_logs(error_res, hard_responses)
        
        structures_gathered_dedup = []
        parents_gathered = []
        
        for struc_list, input_structures in zip(structures_gathered, input_structures_list):
            try:
                dedup_structs = self.remove_duplicate_input(struc_list, input_structures)
            except:
                dedup_structs = struc_list
            # For each structure in the deduped list, record its parents
            for struct in dedup_structs:
               structures_gathered_dedup.append(struct)
               parents_gathered.append(input_structures)  
        return structures_gathered_dedup, parents_gathered

    def save_error_logs(self, error_res, hard_responses):
        # Flatten error results
        errors_flat = []
        for batch_errors in error_res:
            errors_flat.extend(batch_errors)
        
        # Create new DataFrames
        new_error_df = pd.DataFrame(errors_flat, columns=['Error Message', 'Structure String'])
        new_response_df = pd.DataFrame({"Hard Responses": sum(hard_responses, [])})
        
        # Define file paths
        error_path = self.base_path / f'parse_error_{self.args.model_label}_{self.args.save_label}.csv'
        response_path = self.base_path / f'hard_responses_{self.args.model_label}_{self.args.save_label}.csv'
        
        # Concatenate with existing files if they exist
        if os.path.exists(error_path):
            existing_error_df = pd.read_csv(error_path)
            error_df = pd.concat([existing_error_df, new_error_df], ignore_index=True)
        else:
            error_df = new_error_df
            
        if os.path.exists(response_path):
            existing_response_df = pd.read_csv(response_path)
            response_df = pd.concat([existing_response_df, new_response_df], ignore_index=True)
        else:
            response_df = new_response_df
        
        # Save concatenated results
        error_df.to_csv(error_path, index=False)
        response_df.to_csv(response_path, index=False)

    def filter_hypothesis(self, input_data: str) -> Tuple[List[Structure], List[str], List[str]]:
        """Parse structures from JSON string, handling various formats and incomplete inputs."""
        structures: List[Structure] = []
        error_res: List[tuple[str, str]] = []
        hard_responses: List[str] = []
    
        def clean_json(text: str) -> str:
            """Clean JSON string and extract content."""
            text = re.sub(r'```.*?\n|```|\[|\]', '', text)
            text = re.sub(r'["""]', '"', text)
            text = re.sub(r'(\{)(\d+)(:)', r'\1"\2"\3', text)
            return text.strip()
    
        def count_and_fix_composition_poscar(poscar_str: str) -> Optional[str]:
           """Count actual atomic sites and fix composition if needed."""
           try:
               lines = poscar_str.strip().split('\n')
               if len(lines) < 8:
                   return None
               # Basic validation
               try:
                   scale = float(lines[1])
                   lattice = [list(map(float, line.split())) for line in lines[2:5]]
                   if any(len(vec) != 3 for vec in lattice):
                       return None
               except (ValueError, IndexError):
                   return None
               # Get species and validate coordinate type
               species = lines[5].split()
               counts = list(map(int, lines[6].split()))
               if len(species) != len(counts) or not lines[7].lower().startswith(('direct', 'cart')):
                   return None
               # Count actual atoms from coordinates
               actual_composition = {}
               valid_lines = []
               for line in lines[8:]:
                   if not line.strip():
                       continue
                   parts = line.split()
                   if len(parts) < 4:
                       continue
                   try:
                       x, y, z = map(float, parts[:3])
                       element = parts[3]
                       if element in species:  # Only count valid elements
                           actual_composition[element] = actual_composition.get(element, 0) + 1
                           valid_lines.append(line)
                   except (ValueError, IndexError):
                       continue
        
               # Update composition if needed
               if actual_composition:
                   new_species = list(actual_composition.keys())
                   new_counts = [actual_composition[sp] for sp in new_species]
                   
                   # Create new POSCAR with actual composition
                   new_lines = [
                       ' '.join(f"{sp}{cnt}" for sp, cnt in zip(new_species, new_counts)),
                       lines[1],     # Scale factor
                       *lines[2:5],  # Lattice vectors
                       ' '.join(new_species),
                       ' '.join(map(str, new_counts)),
                       lines[7],     # Coordinate type
                       *valid_lines  # Valid coordinates
                   ]
                   return '\n'.join(new_lines)
               return None
        
           except Exception as e:
               print(f"Error in count_and_fix_composition: {str(e)}")
               return None


        def count_and_fix_composition_cif(cif_str: str) -> Optional[str]:
            """Count actual atomic sites and fix composition in CIF format if needed."""
            try:
                # Split CIF into header and loops
                lines = cif_str.strip().split('\n')
                header_lines = []
                loop_lines = []
                current_section = header_lines
                
                for line in lines:
                    if line.strip().startswith('loop_'):
                        current_section = loop_lines
                    elif line.strip():  # Skip empty lines
                        current_section.append(line.strip())
                        
                # Basic validation of cell parameters
                cell_params = {}
                required_params = [
                    '_cell_length_a', '_cell_length_b', '_cell_length_c',
                    '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma'
                ]
                
                for line in header_lines:
                    for param in required_params:
                        if line.startswith(param):
                            try:
                                value = float(line.split()[1])
                                cell_params[param] = value
                            except (ValueError, IndexError):
                                return None
                                
                if len(cell_params) != 6:  # Must have all required parameters
                    return None
                    
                # Parse atom sites
                atom_sites = []
                site_properties = []
                current_property = None
                
                for line in loop_lines:
                    if line.startswith('_atom_site_'):
                        site_properties.append(line.strip())
                    elif not line.startswith('_') and not line.startswith('#'):
                        # This is an atom entry
                        parts = line.split()
                        if len(parts) >= len(site_properties):
                            atom_sites.append(parts)
                            
                if not atom_sites or not site_properties:
                    return None
                    
                # Get indices for required properties
                try:
                    type_idx = site_properties.index('_atom_site_type_symbol')
                    x_idx = site_properties.index('_atom_site_fract_x')
                    y_idx = site_properties.index('_atom_site_fract_y')
                    z_idx = site_properties.index('_atom_site_fract_z')
                except ValueError:
                    return None
                    
                # Count actual composition
                actual_composition = {}
                valid_sites = []
                
                for site in atom_sites:
                    try:
                        element = site[type_idx]
                        x = float(site[x_idx])
                        y = float(site[y_idx])
                        z = float(site[z_idx])
                        
                        # Validate coordinates
                        if all(0 <= coord <= 1 for coord in (x, y, z)):
                            actual_composition[element] = actual_composition.get(element, 0) + 1
                            valid_sites.append(site)
                    except (ValueError, IndexError):
                        continue
                        
                if not actual_composition:
                    return None
                    
                # Reconstruct CIF with actual composition
                formula_parts = [f"{element}{count}" for element, count in actual_composition.items()]
                formula = "".join(formula_parts)
                
                new_cif = [
                    "# generated using pymatgen",
                    f"data_{formula}",
                    "_symmetry_space_group_name_H-M   'P 1'",
                    f"_cell_length_a   {cell_params['_cell_length_a']:.8f}",
                    f"_cell_length_b   {cell_params['_cell_length_b']:.8f}",
                    f"_cell_length_c   {cell_params['_cell_length_c']:.8f}",
                    f"_cell_angle_alpha   {cell_params['_cell_angle_alpha']:.8f}",
                    f"_cell_angle_beta   {cell_params['_cell_angle_beta']:.8f}",
                    f"_cell_angle_gamma   {cell_params['_cell_angle_gamma']:.8f}",
                    "_symmetry_Int_Tables_number   1",
                    f"_chemical_formula_structural   {formula}",
                    f"_chemical_formula_sum   '{' '.join(f'{element}{count}' for element, count in actual_composition.items())}'",
                    "_cell_formula_units_Z   1",
                    "loop_",
                    " _symmetry_equiv_pos_site_id",
                    " _symmetry_equiv_pos_as_xyz",
                    "  1  'x, y, z'",
                    "loop_"
                ]
                
                # Add atom site property headers
                new_cif.extend(f" {prop}" for prop in site_properties)
                
                # Add valid atom sites
                for idx, site in enumerate(valid_sites):
                    site_str = "  " + "  ".join(str(x) for x in site)
                    new_cif.append(site_str)
                    
                return "\n".join(new_cif)
                
            except Exception as e:
                print(f"Error in count_and_fix_composition_cif: {str(e)}")
                return None
        
        def extract_structure_strings(text: str) -> List[str]:
            """Extract and validate structure strings based on format."""
            
            if "crystalllm" in self.args.base_model and self.fmt == 'cif':
                return self.parse_fn(text)
            pattern = r'"' + self.fmt + r'":\s*"([^"]*?)(?:"\s*}|\Z)'
            matches = re.finditer(pattern, text, re.IGNORECASE)
            structure_strings = []
            found_matches = False
            for match in matches:
                found_matches = True
                struct_str = match.group(1).replace('\\n', '\n').strip()
                if self.fmt == 'poscar':
                    if fixed_str := count_and_fix_composition_poscar(struct_str):
                        structure_strings.append(fixed_str)
                    else:
                        error_res.append((f"Invalid {self.fmt.upper()} format", struct_str))
                elif self.fmt == 'cif':
                    if fixed_str := count_and_fix_composition_cif(struct_str):
                        structure_strings.append(fixed_str)
                    else:
                        error_res.append((f"Invalid {self.fmt.upper()} format", struct_str))
                else:
                    return []
            # Try direct structure string if no JSON matches
            if not found_matches:
                if self.fmt == 'poscar' and 'direct' in text.lower():
                    if fixed_str := count_and_fix_composition_poscar(text.strip()):
                        structure_strings.append(fixed_str)
                elif self.fmt == 'cif' and '_cell_length_a' in text:
                    if fixed_str := count_and_fix_composition_cif(text.strip()):
                        structure_strings.append(fixed_str)
            return structure_strings
            
        try:
            cleaned_input = clean_json(input_data)
            structure_strings = extract_structure_strings(cleaned_input)
            print(f"Found {len(structure_strings)} potential {self.fmt.upper()} strings")
            if (len(structure_strings) >= 10):
                print(input_data)
            # Process structures
            for struct_str in structure_strings:
                try:
                    structure = Structure.from_str(struct_str, fmt=self.fmt)
                    print(f"Successfully parsed {self.fmt.upper()} structure: {structure.formula}")
                    structures.append(structure)
                except Exception as e:
                    print(f"Structure parse error: {str(e)}", struct_str)
                    error_res.append((f"Structure parse error: {str(e)}", struct_str))
                    
        except Exception as e:
            print(f"General parsing error: {str(e)}")
            error_res.append((f"General error: {str(e)}", input_data))
            
        if not structures:
            hard_responses.append(input_data)
            print("No structures were successfully parsed")
        
        return structures, error_res, hard_responses



