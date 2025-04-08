##  PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models

![image-20250408210947621](image-20250408210947621.png)

###  Background & Motivation

- **Problem**: Traditional symmetry detection methods (e.g., spatial sampling/PCA) are slow and miss non-orthogonal symmetry planes.
- **Key Insight**: Leverage unsupervised 3D CNN to directly learn symmetry parameters from voxelized shapes.

### Contributions

**PRS-Net Core Innovations**

1. **First Unsupervised DL Solution for Planar Reflective Symmetry Detection**
   - Achieves **100-1000× speedup** compared to state-of-the-art sampling-based methods
   - Processes 3D models in **1.81ms** (vs. 510ms for traditional approaches)
2. **Novel Dual-Loss Framework(two loss functions)**

<u>Symmetry Distance Loss</u> and <u>Regularization Loss</u>

### Key Components:

- **Input**: Voxelized 3D models (32×32×32 resolution)
- **Network**: 5-layer 3D CNN → Fully Connected Layers
- Output
  - **Symmetry Planes**: Normal vector + offset (3 planes)
  - **Rotation Axes**: Quaternion representation (3 axes)

### Loss Functions:

1. **Symmetry Distance Loss (L_sd)**
    Measures deviation between original shape and reflected/rotated counterparts.

   - For reflection: 
     $$
     q'_k = q_k - 2(q_k·n_i + d_i) n_i
     $$
     

   - For rotation: Quaternion transformation

   - Computes shortest distance between transformed points and original surface

2. **Regularization Loss (L_r)**
    Prevents duplicated outputs by enforcing orthogonality:
   $$
   L_r = ||M1M1^T - I||_F^2 + ||M2M2^T - I||_F^2
   $$

3. **Validation Stage**
    Filters invalid/duplicated outputs using:

   - Dihedral angle threshold (π/6)
   - Symmetry error threshold (4×10^-4)
