import numpy as np
from typing import List, Tuple, Dict, Optional
from functools import wraps
import torch
from dataclasses import dataclass
from utils.basic_eval import timeout, TimeoutError
from utils.e_hull_calculator import EHullCalculator
from pymatgen.core.structure import Structure
from chgnet.model import CHGNet
from chgnet.model import StructOptimizer
from chgnet.model.dynamics import EquationOfState

@dataclass
class StabilityResult:
    energy: float = np.inf
    energy_relaxed: float = np.inf
    delta_e: float = np.inf
    e_hull_distance: float = np.inf
    bulk_modulus: float = -np.inf
    bulk_modulus_relaxed: float = -np.inf
    structure_relaxed: Optional[Structure] = None

class StabilityCalculator:
    def __init__(self, mlip="chgnet", ppd_path=""):
        self.mlip = mlip
        self.e_hull = EHullCalculator(ppd_path)
        from pymatgen.io.ase import AseAtomsAdaptor
        self.adaptor = AseAtomsAdaptor()
        self.chgnet = CHGNet.load()
        self.relaxer = StructOptimizer()
        self.EquationOfState = EquationOfState
        if self.mlip == "orb-v3":
            import ase
            from ase.io import read
            from ase import Atoms
            from ase.build import bulk
            from ase.optimize import BFGS
            from ase.eos import EquationOfState as ASE_EquationOfState
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.orb_v3 = pretrained.orb_v3_conservative_inf_omat(
                device=device,
                precision="float32-high"
            )
            self.calculator = ORBCalculator(self.orb_v3, device=device)
            self.ASE_EquationOfState = ASE_EquationOfState
        elif self.mlip == "sevenet":
            from sevenn.calculator import SevenNetD3Calculator
            self.calculator = SevenNetD3Calculator('7net-mf-ompa', modal='omat24', device='cuda')

            
    def _pymatgen_to_ase(self, structure):
        """Convert pymatgen Structure to ASE Atoms"""
        from ase import Atoms
        atoms = Atoms(
            symbols=[site.specie.symbol for site in structure],
            positions=[site.coords for site in structure],
            cell=structure.lattice.matrix, pbc=True
        )
        return atoms
        
    def compute_stability(self, structures: List[Structure], wo_ehull=False, wo_bulk=True) -> Tuple[List[float], List[float]]:
        """Compute stability metrics for a list of structures."""
        results = []
        for structure in structures:
            result = self.process_single_structure(structure, wo_ehull=wo_ehull, wo_bulk=wo_bulk)
            results.append(result)
        return results
            
    def process_single_structure(self, structure: Structure, wo_ehull=False, wo_bulk=True) -> Optional[StabilityResult]:
        """Process single structure stability with error handling."""
        if structure.composition.num_atoms == 0:
            return None
        try:
            relaxation = self.relax_structure(structure, mlip="chgnet")
            if not relaxation or not relaxation['final_structure']:
                return None
            energies = relaxation['trajectory'].energies if hasattr(relaxation['trajectory'], 'energies') else relaxation['trajectory']['energies']
            initial_energy = energies[0] 
            final_energy = energies[-1] # not per atom
            final_structure = relaxation['final_structure']
            # energy_relaxed = self.compute_energy_per_atom(structure_relaxed)
            e_hull_distance = None if wo_ehull else self.compute_ehull_dist(final_structure, final_energy) 
            bulk_modulus = None if wo_bulk else self.compute_bulk_modulus(structure)
            bulk_modulus_relaxed = None if wo_bulk else self.compute_bulk_modulus(final_structure)

            if self.mlip in ["orb-v3", "sevenet"]:
                csp_relaxation = self.relax_structure(structure, mlip=self.mlip)
                if not csp_relaxation or not csp_relaxation['final_structure']:
                    return None
                csp_energies = csp_relaxation['trajectory'].energies if hasattr(csp_relaxation['trajectory'], 'energies') else csp_relaxation['trajectory']['energies']
                initial_energy = csp_energies[0] # overwrite
                final_energy = csp_energies[-1]
            
            initial_energy = initial_energy / structure.num_sites
            final_energy = final_energy / structure.num_sites
            delta_e = final_energy - initial_energy if final_energy is not None else None
            
            return StabilityResult(
                energy=initial_energy,
                e_hull_distance=e_hull_distance,
                delta_e=delta_e,
                bulk_modulus=bulk_modulus,
                energy_relaxed=final_energy,
                bulk_modulus_relaxed=bulk_modulus_relaxed,
                structure_relaxed=final_structure
            )
        except Exception as e:
            print(f"Error processing structure: {e}")
            return None


    # @timeout(60, error_message="Energy computation timed out after 60 seconds")
    # def compute_energy(self, structure: Structure) -> Optional[float]:
    #     """Compute structure energy."""
    #     try:
    #         prediction = self.chgnet.predict_structure(structure)
    #         return float(prediction['e'] * structure.num_sites)
    #     except Exception as e:
    #         print(f"Energy computation error: {e}")
    #         return None

    @timeout(60, error_message="Energy per atom computation timed out after 60 seconds")
    def compute_energy_per_atom(self, structure: Structure) -> Optional[float]:
        """Compute structure energy (per atom)."""
        try:
            if self.mlip == "chgnet":
                prediction = self.chgnet.predict_structure(structure)
                return float(prediction['e'])
            elif self.mlip in ["orb-v3", "sevenet"]:
                atoms = self._pymatgen_to_ase(structure)
                atoms.calc = self.calculator
                crystal_energy = atoms.get_potential_energy()
                return crystal_energy / structure.num_sites
        except Exception as e:
            print(f"Energy per atom computation error: {e}")
            return None


    @timeout(120, error_message="Relaxation timed out after 120 seconds")
    def relax_structure(self, structure: Structure, mlip="chgnet") -> Optional[Dict]:
        """Relax structure with timeout."""
        try:
            if mlip == "chgnet":
                return self.relaxer.relax(structure)
            elif mlip in ["orb-v3", "sevenet"]:
                from ase.optimize import BFGS
                # Convert to ASE Atoms
                atoms = self._pymatgen_to_ase(structure)
                atoms.calc = self.calculator
                
                # Create a basic trajectory to store energies
                trajectory = {'energies': []}
                
                # Store initial energy
                initial_energy = atoms.get_potential_energy()
                trajectory['energies'].append(initial_energy)
                
                # Perform relaxation with BFGS
                optimizer = BFGS(atoms)
                optimizer.run(fmax=0.05, steps=100)
                
                # Store final energy
                final_energy = atoms.get_potential_energy()
                trajectory['energies'].append(final_energy)
                
                final_structure = self.adaptor.get_structure(atoms)
                return {
                        'final_structure': final_structure,
                        'trajectory': trajectory
                    }
            else:
                raise ValueError(f"Unknown MLIP: {self.mlip}")
        except Exception as e:
            print(f"Relaxation error: {e}")
            return None

    @timeout(60, error_message="E-hull distance computation timed out after 60 seconds")
    def compute_ehull_dist(self, structure: Structure, energy: float) -> Optional[float]:
        """Compute energy hull distance."""
        try:
            hull_data = [{
                'structure': structure,
                'energy': energy # energy_per_atom * structure.num_sites
            }]
            return self.e_hull.get_e_hull(hull_data)[0]['e_hull']
        except Exception as e:
            print(f"E-hull computation error: {e}")
            return np.inf

    @timeout(60, error_message="bulk modulus computation timed out after 60 seconds")
    def compute_bulk_modulus(self, structure: Structure) -> Optional[float]:
        """Compute bulk modulus."""
        try:
            if self.mlip == "chgnet":
                eos = self.EquationOfState(model=self.chgnet)
                eos.fit(atoms=structure, steps=500, fmax=0.1, verbose=False)
                return eos.get_bulk_modulus(unit="eV/A^3")
            elif self.mlip == "orb-v3":
                atoms = self._pymatgen_to_ase(structure)
                atoms.calc = self.calculator
                volumes = []
                energies = []
                original_volume = atoms.get_volume()
                for scaling_factor in np.linspace(0.94, 1.06, 7):
                    scaled_atoms = atoms.copy()
                    scaled_atoms.set_cell(atoms.get_cell() * scaling_factor**(1/3), scale_atoms=True)
                    scaled_atoms.calc = self.calculator
                    volumes.append(scaled_atoms.get_volume())
                    energies.append(scaled_atoms.get_potential_energy())
                eos = self.ASE_EquationOfState(volumes, energies)
                v0, e0, B = eos.fit()  # B is the bulk modulus in eV/Ang^3
                B_GPa = B * 160.2176621
                return B_GPa
        except Exception as e:
            print(f"Bulk modulus computation error: {e}")
            return 0.0
    
    def check_stability_rate(self, e_hull_distances: List[float], threshold: float = 0.03) -> Dict:
        """Compute stability statistics for given threshold."""
        if not e_hull_distances:
            return {
                f"stable_rate_{threshold}": 0.0,
                f"stable_num_{threshold}": 0,
                "min_ehull_dist": np.inf,
                "avg_ehull_dist": np.inf
            }
        valid_distances = [d for d in e_hull_distances if not (np.isnan(d) or np.isinf(d) or d is None)]
        stabilities = [d < threshold for d in e_hull_distances]
        return {
            f"stable_rate_{threshold}": np.mean(stabilities),
            f"stable_num_{threshold}": sum(stabilities),
            "min_ehull_dist": min(valid_distances) if valid_distances else np.inf,
            "avg_ehull_dist": np.mean(valid_distances) if valid_distances else np.inf
        }

    def check_local_stability(self, delta_e: List[float]) -> Dict:
        """Compute local stability statistics."""
        valid_delta_e = [d for d in delta_e 
                        if not (np.isnan(d) or np.isinf(d) or d is None)]
        return {
            "avg_delta_e": np.mean(valid_delta_e) if valid_delta_e else np.inf
        }