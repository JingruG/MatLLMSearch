"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import glob
import errno
import signal
import torch
import argparse
import warnings
import functools
import itertools
import numpy as np
import pandas as pd
from p_tqdm import p_map
import cloudpickle as pickle

from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from collections import Counter
from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.composition import ElementProperty
import smact
from smact.screening import pauling_test

from utils.eval_util import (
    chemical_symbols, 
    StandardScaler, 
    CompScalerMeans, 
    CompScalerStds
)

# Much of the below code is taken without modification from the original
# CDVAE repo (https://github.com/txie-93/cdvae).
# In some cases, the code has been modified to work with the structure of
# our codebase, but the logic is the same.

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}

NOVELTY_Cutoffs = {
    'mp20': {'struc': 0.1, 'comp': 2.},
}

CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.)

# CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset('magpie')

class TimeoutError(Exception):
    pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

@timeout(5)
def timeout_featurize(structure, i):
    return CrystalNNFingerprint.from_preset("ops").featurize(structure, i)


def load_data(file_path):
    if file_path[-3:] == 'npy':
        data = np.load(file_path, allow_pickle=True).item()
        for k, v in data.items():
            if k == 'input_data_batch':
                for k1, v1 in data[k].items():
                    data[k][k1] = torch.from_numpy(v1)
            else:
                data[k] = torch.from_numpy(v).unsqueeze(0)
    else:
        data = torch.load(file_path)
    return data

def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()

def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps

def compute_cov(
    crys, 
    gt_crys,
    struc_cutoff, 
    comp_cutoff, 
    num_gen_crystals=None
):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)
    gt_struc_fps, gt_comp_fps = filter_fps(gt_struc_fps, gt_comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.mean(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff))# / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict

def compute_novelty(
    crys, 
    gt_crys,
    struc_cutoff, 
    comp_cutoff,
    num_gen_crystals=None
):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)
    gt_struc_fps, gt_comp_fps = filter_fps(gt_struc_fps, gt_comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_precision_dist = struc_pdist.min(axis=1)
    comp_precision_dist = comp_pdist.min(axis=1)

    struc_novelty = np.mean(struc_precision_dist > struc_cutoff)
    comp_novelty = np.mean(comp_precision_dist > comp_cutoff)

    novelty = np.mean(np.logical_or(
        struc_precision_dist > struc_cutoff,
        comp_precision_dist > comp_cutoff))

    metrics_dict = {
        'struc_novelty': struc_novelty,
        'comp_novelty': comp_novelty,
        'novelty': novelty,
    }

    return metrics_dict

class CDVAEGenEval(object):

    def __init__(
        self, 
        pred_crys, 
        gt_cov_crys, 
        gt_novelty_crys,
        n_samples=1, 
        eval_model_name=None
    ):
        self.crys = pred_crys
        self.gt_cov_crys = gt_cov_crys
        self.gt_novelty_crys = gt_novelty_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name

        valid_crys = [c for c in pred_crys if c.valid]
        self.valid_samples = valid_crys
        # if len(valid_crys) >= n_samples:
        #     sampled_indices = np.random.choice(
        #         len(valid_crys), n_samples, replace=False)
        #     self.valid_samples = [valid_crys[i] for i in sampled_indices]
        # else:
        #     raise Exception(
        #         f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}

    # def get_comp_diversity(self):
    #     comp_fps = [c.comp_fp for c in self.valid_samples]
    #     comp_fps = CompScaler.transform(comp_fps)
    #     comp_div = get_fp_pdist(comp_fps)
    #     return {'comp_div': comp_div}

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        N = comp_fps[0].shape[0] if len(comp_fps[0].shape) > 0 else 1
        theoretical_max = (N ** 0.7)  # Using a higher exponent to account for observed data
        normalized_comp_div = comp_div / theoretical_max
        return {'comp_div': normalized_comp_div}
        
    def get_struct_diversity(self):
        return {'struct_div': get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.gt_cov_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_cov_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_cov_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_novelty(self):
        cutoff_dict = NOVELTY_Cutoffs[self.eval_model_name]
        novelty_metrics_dict = compute_novelty(
            self.crys, self.gt_novelty_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return novelty_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_comp_diversity())
        metrics.update(self.get_struct_diversity())
        metrics.update(self.get_density_wdist())
        metrics.update(self.get_num_elem_wdist())
        # metrics.update(self.get_coverage())
        metrics.update(self.get_novelty())
        return metrics


@timeout(20, error_message="Smact validity computing time out after 20 sec.")
def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False

def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True

class Crystal(object):

    def __init__(self, crys_array_dict):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.dict = crys_array_dict

        self.get_structure()
        self.get_composition()
        self.get_validity()

        if self.valid:
            self.get_fingerprints()
        else:
            self.comp_fp = None
            self.struct_fp = None

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = 'unrealistically_small_lattice'

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        try:
            self.comp_valid = smact_validity(self.elems, self.comps)
        except Exception as e:
            print(f"Error evaluating validity: {e}")
            self.comp_valid = False
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [timeout_featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception as e:
            print('Error constructing fingerprint for crystal. ', e)
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def cif_str_to_crystal(cif_str):
    try:
        structure = Structure.from_str(cif_str, fmt="cif")
        crystal = Crystal({
            "frac_coords": structure.frac_coords,
            "atom_types": [chemical_symbols.index(str(x)) for x in structure.species],
            "lengths": np.array(structure.lattice.parameters[:3]),
            "angles": np.array(structure.lattice.parameters[3:])
        })
    except Exception as e:
        print(e)
        # print(cif_str)
        return None
    
    return crystal

@timeout(30, error_message="Structure to crystal conversion timed out after 30 seconds")
def structure_to_crystal(structure):
    try:
        crystal = Crystal({
            "frac_coords": structure.frac_coords,
            "atom_types": [chemical_symbols.index(str(x)) for x in structure.species],
            "lengths": np.array(structure.lattice.parameters[:3]),
            "angles": np.array(structure.lattice.parameters[3:])
        })
    except Exception as e:
        print(e)
        # print(cif_str)
        return None
    
    return crystal
