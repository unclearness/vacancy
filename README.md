# Vacancy: A Voxel Carving implementation in C++
**Vacancy** is a *Voxel Carving* (a.k.a. *Visual Hull* or *Shape from Silhouette*) implementaion in C++. Inputs are 2D silhouettes (binary mask) of target objects, corresponding camera parameters (both intrinsic and extrinsic) and 3D bounding box to roughly specify the position of the objects. Output is the reconstructed 3D model. In addition to naive one, supports KinectFusion like robust TSDF (Truncated Signed Distance Function) fusion.

# Algorithm Overview
<img src="https://raw.githubusercontent.com/wiki/unclearness/vacancy/images/how_it_works.gif" width="640">

- Initialize 3D voxels according to the input bounding box
- Compute SDF (Signed Distance Function/Field) from silhouette in 2D
- Volmetrically fuse 2D SDF values into the 3D voxels by using camera parameters
- Extract explicit 3D mesh from implicit surface representation in the 3D voxels

# Output mesh
Two mesh extraction methods are implemented: voxel and marching cubes. Marching cubes are much better in practice while voxel representation is suitable to understand how the algorithm works.

|voxel|marching cubes|
|---|---|
|<img src="https://raw.githubusercontent.com/wiki/unclearness/vacancy/images/bunny_voxel.png" width="320">|<img src="https://raw.githubusercontent.com/wiki/unclearness/vacancy/images/bunny_marching_cubes.png" width="320">|

# Build
To build sample bunny executable, use cmake with `CMakeLists.txt` in the top directory.
You can integrate **Vacancy** to your own projects as static library by cmake `add_subdirectory()` command.

# Dependencies
## Mandatory
- Eigen
    https://github.com/eigenteam/eigen-git-mirror
    - Math
## Optional (can be disabled by cmake)
- stb
    https://github.com/nothings/stb
    - Image I/O
- OpenMP (if supported by your compiler)
    - Multi-thread accelaration
