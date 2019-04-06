/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "vacancy/camera.h"
#include "vacancy/common.h"
#include "vacancy/image.h"
#include "vacancy/mesh.h"

namespace vacancy {

// Voxel update type
enum class VoxelUpdate {
  kMin = 0,             // take mininum
  kAverage = 1,         // Average
  kWeightedAverage = 2  // Weighted Average like KinectFusion
};

struct VoxelUpdateOption {
  VoxelUpdate voxel_update{VoxelUpdate::kMin};
  int voxel_max_update_num{
      255};  // After updating voxel_max_update_num, no sdf update
  float voxel_update_weight{1.0f};  // only valid if kWeightedAverage is set
  float truncation_band{0.1f};
};

struct VoxelCarverOption {
  Eigen::Vector3f bb_max;
  Eigen::Vector3f bb_min;
  float resolution{0.001f};

  VoxelUpdateOption update_option;
};

class VoxelGrid;

class VoxelCarver {
  VoxelCarverOption option_;
  std::unique_ptr<VoxelGrid> voxel_grid_;

  void UpdateOnSurface();
  void UpdateOnSurfaceWithPseudo();

 public:
  VoxelCarver();
  ~VoxelCarver();
  VoxelCarver(VoxelCarverOption option);
  void set_option(VoxelCarverOption option);
  void Init();
  bool Carve(const Camera& camera, const Image1b& silhouette);
  bool Carve(const std::vector<Camera>& cameras,
             const std::vector<Image1b>& silhouettes);
  void ExtractVoxel(Mesh* mesh, bool inside_empty = true,
                    bool with_pseudo_surface = false);
  void ExtractIsoSurface(Mesh* mesh);
};

}  // namespace vacancy
