"""
Example usage of the cICA package

This example demonstrates how to use the main cICA functions
to perform contrastive ICA analysis.
"""

import numpy as np
from cica import (
    recover_pattern_tensor_eigen,
    HTD,
    cumulant_tensors,
    subspace_power_method
)

# Example 1: Using cICA with synthetic data
def example_synthetic_data():
    """
    Example of applying cICA to synthetic foreground and background data
    """
    print("Example 1: Synthetic data")
    print("-" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples_fg = 1000
    n_samples_bg = 1000
    n_features = 10
    
    # Foreground data (signal of interest)
    X_foreground = np.random.randn(n_samples_fg, n_features)
    
    # Background data (confounding signal)
    X_background = np.random.randn(n_samples_bg, n_features)
    
    # Compute cumulant tensors
    print("Computing cumulant tensors...")
    k2_f, k4_f = cumulant_tensors(X_foreground)
    k2_b, k4_b = cumulant_tensors(X_background)
    
    # Apply cICA via SPM then HTD
    print("Applying cICA (SPM + HTD)...")
    a_s, b_s, contrast_values = recover_pattern_tensor_eigen(
        k4_b, k4_f, k2_f, k2_b, 
        r=None,  # Automatically determine rank
        l=None,  # Automatically determine number of foreground components
        verbose=True
    )
    
    print(f"\nRecovered {a_s.shape[1]} background patterns")
    print(f"Recovered {b_s.shape[1]} foreground patterns")
    print(f"Top 3 contrast values: {contrast_values[:3]}")
    print()
    

# Example 2: Using proportional cICA (HTD only)
def example_proportional_cica():
    """
    Example of applying proportional cICA (HTD method)
    """
    print("Example 2: Proportional cICA (HTD)")
    print("-" * 50)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples_fg = 1000
    n_samples_bg = 1000
    n_features = 8
    
    X_foreground = np.random.randn(n_samples_fg, n_features)
    X_background = np.random.randn(n_samples_bg, n_features)
    
    # Compute cumulant tensors
    print("Computing cumulant tensors...")
    k2_f, k4_f = cumulant_tensors(X_foreground)
    k2_b, k4_b = cumulant_tensors(X_background)
    
    # Apply proportional cICA
    print("Applying proportional cICA...")
    gamma = 1.0  # Proportionality constant
    l = 5  # Number of components to extract
    
    b_s, contrast_values = HTD(k4_b, k4_f, k2_b, k2_f, gamma, l)
    
    print(f"Recovered {b_s.shape[1]} components")
    print(f"Contrast values: {contrast_values}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("cICA Package Examples")
    print("=" * 50)
    print()
    
    # Run examples
    example_synthetic_data()
    example_proportional_cica()
    
    print("=" * 50)
    print("Examples completed successfully!")
    print("=" * 50)
