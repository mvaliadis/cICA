# cICA Package Documentation

## Overview

The cICA (Contrastive Independent Component Analysis) package provides implementations of contrastive ICA algorithms for identifying independent components that are distinctive in a foreground dataset compared to a background dataset.

## Main Functions

### 1. `cumulant_tensors(observeddata)`

Computes the second and fourth order cumulant tensors from observed data.

**Parameters:**
- `observeddata`: numpy array of shape (n_samples, n_features) - The observed data matrix

**Returns:**
- `k2`: Second-order cumulant tensor (covariance matrix)
- `k4`: Fourth-order cumulant tensor

**Example:**
```python
import numpy as np
from cica import cumulant_tensors

data = np.random.randn(1000, 10)
k2, k4 = cumulant_tensors(data)
```

### 2. `recover_pattern_tensor_eigen(k4_b, k4_f, k2_f, k2_b, r=None, l=None, verbose=False, step_max=1)`

Main cICA algorithm using SPM (Subspace Power Method) followed by HTD (Higher-order Tensor Decomposition).

**Parameters:**
- `k4_b`: Fourth-order cumulant tensor of background data
- `k4_f`: Fourth-order cumulant tensor of foreground data
- `k2_f`: Second-order cumulant tensor (covariance) of foreground data
- `k2_b`: Second-order cumulant tensor (covariance) of background data
- `r`: Number of background patterns to extract (default: auto-determined)
- `l`: Number of foreground patterns to extract (default: auto-determined)
- `verbose`: Whether to print progress information
- `step_max`: Maximum number of steps for preventing repetitive vectors

**Returns:**
- `a_s`: Background patterns (numpy array of shape [n_features, r])
- `b_s`: Foreground patterns sorted by contrast (numpy array of shape [n_features, l])
- `contrast_values`: Contrast values for each foreground pattern (higher is better)

**Example:**
```python
from cica import recover_pattern_tensor_eigen, cumulant_tensors

# Compute cumulant tensors
k2_f, k4_f = cumulant_tensors(foreground_data)
k2_b, k4_b = cumulant_tensors(background_data)

# Apply cICA
a_s, b_s, contrast = recover_pattern_tensor_eigen(
    k4_b, k4_f, k2_f, k2_b, 
    verbose=True
)

# The columns of b_s are the foreground patterns,
# sorted by contrast (most contrastive first)
top_pattern = b_s[:, 0]
```

### 3. `HTD(k4_b, k4_f, k2_b, k2_f, gamma, l)`

Proportional cICA algorithm using only Higher-order Tensor Decomposition (HTD).

**Parameters:**
- `k4_b`: Fourth-order cumulant tensor of background data
- `k4_f`: Fourth-order cumulant tensor of foreground data
- `k2_b`: Second-order cumulant tensor of background data
- `k2_f`: Second-order cumulant tensor of foreground data
- `gamma`: Proportionality constant for contrasting
- `l`: Number of components to extract

**Returns:**
- `b_s`: Extracted patterns sorted by contrast
- `contrast_values`: Contrast values for each pattern

**Example:**
```python
from cica import HTD, cumulant_tensors

k2_f, k4_f = cumulant_tensors(foreground_data)
k2_b, k4_b = cumulant_tensors(background_data)

b_s, contrast = HTD(k4_b, k4_f, k2_b, k2_f, gamma=1.0, l=5)
```

### 4. `HTD_decomp(T, l)`

Pure HTD tensor decomposition without contrast.

**Parameters:**
- `T`: Input tensor to decompose
- `l`: Number of components to extract

**Returns:**
- `blist`: Extracted patterns

### 5. `subspace_power_method(T, d=None, n=None, r=None, **kwargs)`

Symmetric tensor decomposition using the Subspace Power Method.

**Parameters:**
- `T`: Input symmetric tensor
- `d`: Dimension (auto-determined from T if not provided)
- `n`: Order of the tensor (auto-determined from T if not provided)
- `r`: Rank of the decomposition (auto-determined if not provided)
- `**kwargs`: Additional options (maxiter, ntries, gradtol, eigtol, ftol, w_out)

**Returns:**
- `A`: Matrix of decomposed components
- `w`: Weights (if w_out=True)

## Typical Workflow

1. **Prepare your data**: Organize your data into foreground and background datasets
2. **Compute cumulant tensors**: Use `cumulant_tensors()` on both datasets
3. **Apply cICA**: Use `recover_pattern_tensor_eigen()` for the full algorithm or `HTD()` for proportional cICA
4. **Analyze results**: The returned patterns are sorted by contrast, with the most distinctive patterns first

## References

- SPM implementation based on: "Subspace power method for symmetric tensor decomposition and generalized PCA" by Joe Kileel and João M. Pereira
