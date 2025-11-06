# cICA
This repo contains code for contrastive ICA algorithms and code for comparing cICA algorithms to other algorithms in various datasets.

SPM.py, helper_functions.py, compiler_options.py are code from the paper 'Subspace power method for symmetric tensor decomposition and generalized PCA' by Joe Kileel and João M. Pereira.


The code for cICA algorithms is in the file cICA_functions.py.


The code for comparing cICA algorithms to other contrastive algorithms in various datasets is in the file numerical_experiments.ipynb.
The code for pre-processing the monkey-human dataset provided by the paper 'Comparative single-cell transcriptomic analysis of primate brains highlights human-specific regulatory evolution' is in monkey_human.r and the code for applying cICA to the processed data is in monkey_human.ipynb.

The code for justifying our choice of applying SPM then HTD for the three step coupled tensor decomposition is in experiments_justify_SPM_HTD.ipynb. The other possible combinations of ICA or tensor decomposition algorithms is in cICA_alternative_tensor_decomps.py. 


Data availability:
We upload the mouse protein data from the paper 'Self-organizing feature maps identify proteins critical to learning in a mouse model of down syndrome'.
And the preprocessed datasets for other real-world datasets. 
