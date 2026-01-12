# cICA

A Python package for contrastive Independent Component Analysis (cICA) algorithms.

This repository contains code for contrastive ICA algorithms and code for comparing cICA algorithms to other algorithms in various datasets.

## Installation

You can install the package directly from the repository:

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from cica import recover_pattern_tensor_eigen, cumulant_tensors

# Load your foreground and background data
X_foreground = np.random.randn(1000, 10)  # Your foreground data
X_background = np.random.randn(1000, 10)  # Your background data

# Compute cumulant tensors
k2_f, k4_f = cumulant_tensors(X_foreground)
k2_b, k4_b = cumulant_tensors(X_background)

# Apply cICA
a_s, b_s, contrast_values = recover_pattern_tensor_eigen(
    k4_b, k4_f, k2_f, k2_b, 
    r=None,  # Automatically determine rank
    l=None,  # Automatically determine number of components
    verbose=True
)

print(f"Recovered {b_s.shape[1]} foreground patterns")
print(f"Top contrast values: {contrast_values[:3]}")
```

See `example_usage.py` for more detailed examples.

## Package Structure

- `cica/cICA_functions.py` - Main cICA algorithms (HTD, SPM+HTD)
- `cica/ICA_code.py` - Cumulant tensor computation
- `cica/SPM.py` - Subspace Power Method implementation
- `cica/helper_functions.py` - Helper functions for tensor operations
- `cica/cICA_alternative_tensor_decomps.py` - Alternative tensor decomposition methods

## Credits

SPM.py, helper_functions.py, compiler_options.py are code from the paper 'Subspace power method for symmetric tensor decomposition and generalized PCA' by Joe Kileel and João M. Pereira.


The code for cICA algorithms is in the file cICA_functions.py.


The code for comparing cICA algorithms to other contrastive algorithms in various datasets is in the file numerical_experiments.ipynb.
The code for pre-processing the monkey-human dataset provided by the paper 'Comparative single-cell transcriptomic analysis of primate brains highlights human-specific regulatory evolution' is in monkey_human.r and the code for applying cICA to the processed data is in monkey_human.ipynb.

The code for justifying our choice of applying SPM then HTD for the three step coupled tensor decomposition is in experiments_justify_SPM_HTD.ipynb. The other possible combinations of ICA or tensor decomposition algorithms is in cICA_alternative_tensor_decomps.py. 


Data availability:
We upload the mouse protein data from the paper 'Self-organizing feature maps identify proteins critical to learning in a mouse model of down syndrome' and the preprocessed datasets for other real-world datasets. 
