# Self-Supervised SOL
This is the code for the paper "Data-driven and self-supervised spectral operator learning methods for the heat condution equation with variable source functions".

-'poisson' contains the codebase and trained SOL models.

-'SOL2_0' contains the necessary source code of the SOL models, and we refer to [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) and [SPFNO](https://github.com/liu-ziyuan-math/SPFNO) for details.

## Dataset
The dataset we used in the experiments are now available at 
-[Data](https://drive.google.com/drive/folders/1A49PPsLGTFS0vUuBg0_pfZlDEZPPZqLG?usp=sharing)

which contains both the datasets for the Poisson equation under homogeneous Dirichlet conditions and mixed BCs.

Each file is loaded as tensors, in the form of (size, Nx, Ny), with size = 1000, Nx = Ny = 401.
