"""
cICA - Contrastive Independent Component Analysis

A Python package for contrastive ICA algorithms.
"""

from .cICA_functions import (
    HTD,
    HTD_decomp,
    recover_pattern_tensor_eigen,
    similarity_measures_withpermutation,
    eig2,
    return_residual
)

from .ICA_code import cumulant_tensors

from .SPM import subspace_power_method, generate_lowrank_tensor

from .helper_functions import (
    khatri_rao_product,
    khatri_rao_power,
    tucker_product,
    symmetric_indices,
    generate_lowrank_tensor_general
)

__version__ = "0.1.0"

__all__ = [
    # Main cICA functions
    "HTD",
    "HTD_decomp", 
    "recover_pattern_tensor_eigen",
    "similarity_measures_withpermutation",
    "cumulant_tensors",
    "subspace_power_method",
    # Helper functions
    "khatri_rao_product",
    "khatri_rao_power",
    "tucker_product",
    "symmetric_indices",
    "generate_lowrank_tensor",
    "generate_lowrank_tensor_general",
    "eig2",
    "return_residual"
]
