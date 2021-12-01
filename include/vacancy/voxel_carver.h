/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "vacancy/camera.h"
#include "vacancy/common.h"
#include "vacancy/image.h"
#include "vacancy/mesh.h"

namespace vacancy {

// Voxel update type
enum class VoxelUpdate {
  kMax = 0,             // take max. naive voxel carving
  kWeightedAverage = 1  // weighted average like KinectFusion. truncation is
                        // necessary to get good result
};

// Interpolation method for 2D SDF
enum class SdfInterpolation {
  kNn = 0,       // Nearest Neigbor
  kBilinear = 1  // Bilinear interpolation
};

// The way to update voxels which are projected to outside of the current image
enum class UpdateOutsideImage {
  kNone = 0,  // Do nothing
  kMax = 1    // Fill by max sdf of the current image. This is valid only when
              // silhouette is not protruding over the current image edge
};

struct InvalidSdf {
  static const float kVal;
};

struct VoxelUpdateOption {
  VoxelUpdate voxel_update{VoxelUpdate::kMax};
  SdfInterpolation sdf_interp{SdfInterpolation::kBilinear};
  UpdateOutsideImage update_outside{UpdateOutsideImage::kNone};
  int voxel_max_update_num{
      255};  // After updating voxel_max_update_num, no sdf update
  float voxel_update_weight{1.0f};  // only valid if kWeightedAverage is set
  bool use_truncation{false};
  float truncation_band{0.1f};  // only positive value is valid
};

struct VoxelCarverOption {
  Eigen::Vector3f bb_max;
  Eigen::Vector3f bb_min;
  float resolution{0.1f};  // default is 10cm if input is m-scale
  bool sdf_minmax_normalize{true};
  VoxelUpdateOption update_option;
};

struct Voxel {
  Eigen::Vector3i index{-1, -1, -1};  // voxel index
  int id{-1};
  Eigen::Vector3f pos{0.0f, 0.0f, 0.0f};  // center of voxel
  float sdf{0.0f};  // Signed Distance Function (SDF) value
  int update_num{0};
  bool outside{false};
  bool on_surface{false};
  Voxel();
  ~Voxel();
};

class VoxelGrid {
  std::vector<Voxel> voxels_;
  Eigen::Vector3f bb_max_;
  Eigen::Vector3f bb_min_;
  float resolution_{-1.0f};
  Eigen::Vector3i voxel_num_{0, 0, 0};
  int xy_slice_num_{0};

 public:
  VoxelGrid();
  ~VoxelGrid();
  bool Init(const Eigen::Vector3f& bb_max, const Eigen::Vector3f& bb_min,
            float resolution);
  const Eigen::Vector3i& voxel_num() const;
  const Voxel& get(int x, int y, int z) const;
  Voxel* get_ptr(int x, int y, int z);
  float resolution() const;
  void ResetOnSurface();
  bool initialized() const;
};

class VoxelCarver {
  VoxelCarverOption option_;
  std::unique_ptr<VoxelGrid> voxel_grid_;

 public:
  VoxelCarver();
  ~VoxelCarver();
  explicit VoxelCarver(VoxelCarverOption option);
  void set_option(VoxelCarverOption option);
  bool Init();
  bool Carve(const Camera& camera, const Image1b& silhouette,
             const Eigen::Vector2i& roi_min, const Eigen::Vector2i& roi_max,
             Image1f* sdf);
  bool Carve(const Camera& camera, const Eigen::Vector2i& roi_min,
             const Eigen::Vector2i& roi_max, const Image1f& sdf);
  bool Carve(const Camera& camera, const Image1b& silhouette, Image1f* sdf);
  bool Carve(const Camera& camera, const Image1b& silhouette);
  bool Carve(const Camera& camera, const Image1f& sdf);
  bool Carve(const std::vector<Camera>& cameras,
             const std::vector<Image1b>& silhouettes);
  void ExtractVoxel(Mesh* mesh, bool inside_empty = false);
  void ExtractIsoSurface(Mesh* mesh, double iso_level = 0.0,
                         bool linear_interp = true);
};

void DistanceTransformL1(const Image1b& mask, const Eigen::Vector2i& roi_min,
                         const Eigen::Vector2i& roi_max, Image1f* dist);
void MakeSignedDistanceField(const Image1b& mask,
                             const Eigen::Vector2i& roi_min,
                             const Eigen::Vector2i& roi_max, Image1f* dist,
                             bool minmax_normalize, bool use_truncation,
                             float truncation_band);
void SignedDistance2Color(const Image1f& sdf, Image3b* vis_sdf,
                          float min_negative_d, float max_positive_d);

}  // namespace vacancy
