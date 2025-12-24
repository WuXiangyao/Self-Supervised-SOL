# Self-Supervised Spectral Operator Learning
This is the code for the paper "Data-driven and self-supervised spectral operator learning methods for the heat condution equation with variable source functions".

- 'poisson' contains the codebase.

- 'poisson/poisson.py','poisson/poisson_mixedBC.py': code for training the SOL models addressing the solution operator of the Poisson equation under homogeneous Dirichlet boundary conditions and mixed BCs.

- 'poisson/SOL2_0' contains the necessary source code of the SOL models, and we refer to [Orthogonal Polynomial Neural Operator, OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) and [semi-periodic Fourier neural operator, SPFNO](https://github.com/liu-ziyuan-math/SPFNO) for details.

- We would upload the trained models in 'poisson/models' soon.

## Dataset
The dataset we used in the experiments are now available at 

- [Data](https://drive.google.com/drive/folders/1A49PPsLGTFS0vUuBg0_pfZlDEZPPZqLG?usp=sharing)

which contains both the datasets for the Poisson equation under homogeneous Dirichlet BCs and mixed BCs.

Each file is loaded as tensors, in the form of (size, Nx, Ny), with size = 1000, Nx = Ny = 401.
