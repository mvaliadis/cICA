# Installation Guide

## Quick Installation

Install the cICA package directly from the repository:

```bash
pip install -e .
```

## Installation with Development Tools

If you plan to contribute or modify the code:

```bash
pip install -e ".[dev]"
```

This will also install Jupyter, matplotlib, and pandas for running the example notebooks.

## Verify Installation

After installation, verify that the package is working:

```python
python -c "import cica; print('cICA version:', cica.__version__)"
```

## System Requirements

- Python 3.7 or higher
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- scikit-learn >= 0.24.0
- PyMoments >= 0.1.0

## Optional Dependencies

For improved performance with numba:
```bash
pip install numba
```

## Troubleshooting

### PyMoments Installation Issues

If you encounter issues installing PyMoments, you can install it separately:

```bash
pip install PyMoments
```

### Import Errors

If you get import errors, make sure you're in a fresh Python session after installation:

```bash
# Close any existing Python sessions
python -c "import cica"
```

## Uninstallation

To remove the package:

```bash
pip uninstall cica
```

## For Developers

If you're developing the package, install in editable mode with development dependencies:

```bash
git clone https://github.com/mvaliadis/cICA.git
cd cICA
pip install -e ".[dev]"
```

This allows you to modify the code and see changes immediately without reinstalling.
