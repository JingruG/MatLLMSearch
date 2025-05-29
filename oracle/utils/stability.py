import numpy as np
from typing import List, Tuple, Dict, Optional
from functools import wraps
import torch
from dataclasses import dataclass
from utils.e_hull_calculator import EHullCalculator
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from chgnet.graph import CrystalGraphConverter
from chgnet.model import CHGNet, StructOptimizer
from chgnet.model.dynamics import EquationOfState
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time
import copy

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
    def __init__(self, mlip="chgnet", ppd_path="", device="cuda"):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.mlip = mlip
        self.ppd_path = ppd_path  # Store for multi-GPU workers
        self.e_hull = EHullCalculator(ppd_path)
        self.adaptor = AseAtomsAdaptor()
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        # Initialize models exactly as in original
        self.chgnet = CHGNet.load().to(self.device)
        converter = CrystalGraphConverter(
            atom_graph_cutoff=6, bond_graph_cutoff=3, algorithm="fast", on_isolated_atoms="warn"
        )
        self.chgnet.graph_converter = converter
        self.relaxer = StructOptimizer(model=self.chgnet, use_device='cuda:0')
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
        
        print(f"Available GPUs: {self.num_gpus}")

    def _safe_timeout_wrapper(self, func, timeout_seconds, *args, **kwargs):
        """Thread-safe timeout wrapper that doesn't use signals"""
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                print(f"Operation timed out after {timeout_seconds} seconds")
                return None
            except Exception as e:
                print(f"Operation failed: {e}")
                return None

    def _pymatgen_to_ase(self, structure):
        """Convert pymatgen Structure to ASE Atoms"""
        from ase import Atoms
        atoms = Atoms(
            symbols=[site.specie.symbol for site in structure],
            positions=[site.coords for site in structure],
            cell=structure.lattice.matrix, pbc=True
        )
        return atoms

    def compute_stability(self, structures: List[Structure], wo_ehull=False, wo_bulk=True) -> List[Optional[StabilityResult]]:
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
            bulk_modulus = None
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

    def compute_energy_per_atom(self, structure: Structure) -> Optional[float]:
        """Compute structure energy (per atom). SAFE TIMEOUT VERSION."""
        def _compute_energy():
            if self.mlip == "chgnet":
                prediction = self.chgnet.predict_structure(structure)
                return float(prediction['e'])
            elif self.mlip in ["orb-v3", "sevenet"]:
                atoms = self._pymatgen_to_ase(structure)
                atoms.calc = self.calculator
                crystal_energy = atoms.get_potential_energy()
                return crystal_energy / structure.num_sites
        
        try:
            return self._safe_timeout_wrapper(_compute_energy, 60)
        except Exception as e:
            print(f"Energy per atom computation error: {e}")
            return None

    def relax_structure(self, structure: Structure, mlip="chgnet") -> Optional[Dict]:
        """Relax structure with safe timeout."""
        def _relax_chgnet():
            return self.relaxer.relax(structure)
        
        def _relax_ase():
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
        
        try:
            if mlip == "chgnet":
                return self._safe_timeout_wrapper(_relax_chgnet, 120)
            elif mlip in ["orb-v3", "sevenet"]:
                return self._safe_timeout_wrapper(_relax_ase, 120)
            else:
                raise ValueError(f"Unknown MLIP: {mlip}")
        except Exception as e:
            print(f"Relaxation error: {e}")
            return None

    def compute_ehull_dist(self, structure: Structure, energy: float) -> Optional[float]:
        """Compute energy hull distance. SAFE TIMEOUT VERSION."""
        def _compute_ehull():
            hull_data = [{
                'structure': structure,
                'energy': energy # energy_per_atom * structure.num_sites
            }]
            return self.e_hull.get_e_hull(hull_data)[0]['e_hull']
        
        try:
            result = self._safe_timeout_wrapper(_compute_ehull, 60)
            return result if result is not None else np.inf
        except Exception as e:
            print(f"E-hull computation error: {e}")
            return np.inf

    def compute_bulk_modulus(self, structure: Structure) -> Optional[float]:
        """Compute bulk modulus. SAFE TIMEOUT VERSION."""
        def _compute_bulk_chgnet():
            eos = self.EquationOfState(model=self.chgnet)
            eos.fit(atoms=structure, steps=500, fmax=0.1, verbose=False)
            return eos.get_bulk_modulus(unit="eV/A^3")
        
        def _compute_bulk_orb():
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
        
        try:
            if self.mlip == "chgnet":
                result = self._safe_timeout_wrapper(_compute_bulk_chgnet, 60)
            elif self.mlip == "orb-v3":
                result = self._safe_timeout_wrapper(_compute_bulk_orb, 60)
            else:
                result = 0.0
            
            return result if result is not None else 0.0
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

    # ============ MULTI-GPU METHODS ============
    
    def compute_stability_parallel(self, structures: List[Structure], wo_ehull=False, wo_bulk=True, 
                                 num_workers=None) -> List[Optional[StabilityResult]]:
        """
        Multi-GPU parallel version using safe timeouts
        """
        if len(structures) == 0:
            return []
            
        if num_workers is None:
            num_workers = min(self.num_gpus, len(structures), 4)
            
        print(f"Processing {len(structures)} structures using {num_workers} workers")
        
        results = [None] * len(structures)
        
        def worker_function(worker_id, structure_batch):
            """Worker function that processes a batch of structures"""
            device_str = f"cuda:{worker_id % self.num_gpus}" if torch.cuda.is_available() else "cpu"
            worker_results = {}
            
            try:
                # Create calculator for this worker with same parameters as original
                worker_calc = StabilityCalculator(
                    mlip=self.mlip, 
                    ppd_path=self.ppd_path, 
                    device=device_str
                )
                
                print(f"Worker {worker_id} starting on {device_str} with {len(structure_batch)} structures")
                
                for original_idx, structure in structure_batch:
                    try:
                        # Use the same process_single_structure (now with safe timeouts)
                        result = worker_calc.process_single_structure(structure, wo_ehull=wo_ehull, wo_bulk=wo_bulk)
                        worker_results[original_idx] = result
                        
                        if (len(worker_results) % 5 == 0):
                            print(f"Worker {worker_id}: processed {len(worker_results)} structures")
                            
                    except Exception as e:
                        print(f"Worker {worker_id}, structure {original_idx}: {str(e)}")
                        worker_results[original_idx] = None
                        
            except Exception as e:
                print(f"Worker {worker_id} failed to initialize: {str(e)}")
                for original_idx, _ in structure_batch:
                    worker_results[original_idx] = None
            
            return worker_results

        # Split structures into batches for each worker
        batch_size = max(1, len(structures) // num_workers)
        structure_batches = []
        
        for i in range(0, len(structures), batch_size):
            batch = [(j, structures[j]) for j in range(i, min(i + batch_size, len(structures)))]
            structure_batches.append(batch)
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_worker = {}
            for worker_id, batch in enumerate(structure_batches):
                if batch:  # Only submit non-empty batches
                    future = executor.submit(worker_function, worker_id, batch)
                    future_to_worker[future] = worker_id
            
            # Collect results
            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    worker_results = future.result()
                    for original_idx, result in worker_results.items():
                        if original_idx < len(results):
                            results[original_idx] = result
                except Exception as e:
                    print(f"Worker {worker_id} failed: {e}")
        
        successful = sum(1 for r in results if r is not None)
        print(f"Processing complete: {successful}/{len(structures)} structures processed successfully")
        
        return results

    def compute_stability_gpu_queues(self, structures: List[Structure], wo_ehull=False, wo_bulk=True) -> List[Optional[StabilityResult]]:
        """
        Queue-based multi-GPU processing using safe timeouts
        """
        if len(structures) == 0:
            return []
            
        print(f"Using {self.num_gpus} GPUs for processing {len(structures)} structures")
        
        # Create queues for distributing work
        structure_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Fill the structure queue
        for i, structure in enumerate(structures):
            structure_queue.put((i, structure))
        
        def gpu_worker(gpu_id):
            """Worker function for each GPU"""
            device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            
            try:
                # Create calculator for this GPU with same parameters as original
                worker_calc = StabilityCalculator(
                    mlip=self.mlip, 
                    ppd_path=self.ppd_path, 
                    device=device_str
                )
                
                print(f"GPU {gpu_id} worker started on {device_str}")
                processed_count = 0
                
                while True:
                    try:
                        # Get work from queue with timeout
                        index, structure = structure_queue.get(timeout=5)
                        
                        try:
                            # Use the same process_single_structure (now with safe timeouts)
                            result = worker_calc.process_single_structure(structure, wo_ehull=wo_ehull, wo_bulk=wo_bulk)
                            result_queue.put((index, result))
                            processed_count += 1
                            
                            if processed_count % 5 == 0:
                                print(f"GPU {gpu_id}: processed {processed_count} structures")
                                
                        except Exception as e:
                            print(f"GPU {gpu_id}: Error processing structure {index}: {e}")
                            result_queue.put((index, None))
                        finally:
                            structure_queue.task_done()
                            
                    except queue.Empty:
                        # No more work available
                        break
                        
            except Exception as e:
                print(f"GPU {gpu_id}: Worker initialization failed: {e}")
        
        # Start worker threads
        threads = []
        for gpu_id in range(self.num_gpus):
            thread = threading.Thread(target=gpu_worker, args=(gpu_id,), daemon=True)
            thread.start()
            threads.append(thread)
        
        # Collect results
        results = [None] * len(structures)
        collected = 0
        
        print("Collecting results...")
        while collected < len(structures):
            try:
                index, result = result_queue.get(timeout=10)
                results[index] = result
                collected += 1
                
                if collected % 10 == 0:
                    print(f"Collected {collected}/{len(structures)} results")
                    
            except queue.Empty:
                print(f"Timeout waiting for results. Collected: {collected}/{len(structures)}")
                alive_threads = [t for t in threads if t.is_alive()]
                if not alive_threads:
                    print("No threads alive, stopping collection")
                    break
        
        # Wait for threads to finish
        for thread in threads:
            thread.join(timeout=30)
        
        successful = sum(1 for r in results if r is not None)
        print(f"Processing complete: {successful}/{len(structures)} structures processed successfully")
        return results

import time

def quick_benchmark(calc, test_structures, wo_ehull=False, wo_bulk=True):
    """
    Quick comparison of all three methods
    """
    print(f"Benchmarking {len(test_structures)} structures...")
    print("=" * 50)
    
    methods = [
        ("Single GPU", calc.compute_stability),
        ("Multi-GPU Parallel", calc.compute_stability_parallel), 
        ("GPU Queues", calc.compute_stability_gpu_queues)
    ]
    
    results = {}
    
    for name, method in methods:
        print(f"\n{name}:")
        print("-" * 20)
        
        start_time = time.time()
        try:
            method_results = method(test_structures, wo_ehull=wo_ehull, wo_bulk=wo_bulk)
            end_time = time.time()
            
            total_time = end_time - start_time
            successful = sum(1 for r in method_results if r is not None)
            success_rate = successful / len(test_structures) * 100
            throughput = successful / total_time * 60  # per minute
            
            results[name] = {
                'time': total_time,
                'successful': successful,
                'success_rate': success_rate,
                'throughput': throughput,
                'time_per_struct': total_time / len(test_structures)
            }
            
            print(f"✓ Total time: {total_time:.1f} seconds")
            print(f"✓ Success rate: {success_rate:.1f}% ({successful}/{len(test_structures)})")
            print(f"✓ Throughput: {throughput:.1f} structures/minute")
            print(f"✓ Time per structure: {total_time/len(test_structures):.2f} seconds")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            results[name] = None
    
    # Summary comparison
    print("\n" + "=" * 50)
    print("SUMMARY COMPARISON:")
    print("=" * 50)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) > 1:
        fastest = min(valid_results.keys(), key=lambda x: valid_results[x]['time'])
        highest_throughput = max(valid_results.keys(), key=lambda x: valid_results[x]['throughput'])
        most_reliable = max(valid_results.keys(), key=lambda x: valid_results[x]['success_rate'])
        
        print(f"🚀 Fastest method: {fastest} ({valid_results[fastest]['time']:.1f}s)")
        print(f"⚡ Highest throughput: {highest_throughput} ({valid_results[highest_throughput]['throughput']:.1f} struct/min)")
        print(f"🎯 Most reliable: {most_reliable} ({valid_results[most_reliable]['success_rate']:.1f}% success)")
        
        # Speed comparison
        if len(valid_results) >= 2:
            times = [v['time'] for v in valid_results.values()]
            speedup = max(times) / min(times)
            print(f"📈 Max speedup: {speedup:.1f}x")
    
    return results

# Usage example:
if __name__ == "__main__":
    # Create calculator exactly as your original
    from pymatgen.core.structure import Structure
    import pandas as pd     
    from utils.stability import StabilityCalculator, quick_benchmark
    calc = StabilityCalculator(mlip="chgnet", ppd_path='resources/2023-02-07-ppd-mp.pkl.gz', device="cuda")

    seed_structures_df = pd.read_csv('../results/poscar_70b_100_2_5_ehull/generations.csv')
    seed_structures_df['e_hull_distance'] = seed_structures_df['EHullDistance']
    seed_structures_df['structure'] = seed_structures_df['Structure']
    seed_structures_df['structure'] = seed_structures_df['structure'].apply(lambda x: Structure.from_str(x, fmt='json') if pd.notna(x) else None)
    structures = seed_structures_df['structure'].tolist()
    test_structures = structures[:40]
    # results = quick_benchmark(calc, test_structures)


    # All methods now use safe timeouts
    single_results = calc.compute_stability(test_structures, wo_ehull=False, wo_bulk=True)
    parallel_results = calc.compute_stability_parallel(test_structures, wo_ehull=False, wo_bulk=True)
    queue_results = calc.compute_stability_gpu_queues(test_structures, wo_ehull=False, wo_bulk=True)

    # Verify results are identical
    for i, (single, parallel, queue_r) in enumerate(zip(single_results, parallel_results, queue_results)):
        if single is not None and parallel is not None and queue_r is not None:
            print(f"Structure {i}:")
            print(f"  Single: energy={single.energy:.6f}, e_hull={single.e_hull_distance:.6f}")
            print(f"  Parallel: energy={parallel.energy:.6f}, e_hull={parallel.e_hull_distance:.6f}")
            print(f"  Queue: energy={queue_r.energy:.6f}, e_hull={queue_r.e_hull_distance:.6f}")
            print(f"  Energy match: {abs(single.energy - parallel.energy) < 1e-6}")
            print(f"  E-hull match: {abs(single.e_hull_distance - parallel.e_hull_distance) < 1e-6}")