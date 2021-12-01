/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "vacancy/voxel_carver.h"

#include <array>

#include "vacancy/extract_voxel.h"
#include "vacancy/marching_cubes.h"
#include "vacancy/timer.h"

namespace {

inline float SdfInterpolationNn(const Eigen::Vector2f& image_p,
                                const vacancy::Image1f& sdf,
                                const Eigen::Vector2i& roi_min,
                                const Eigen::Vector2i& roi_max) {
  Eigen::Vector2i image_p_i(static_cast<int>(std::round(image_p.x())),
                            static_cast<int>(std::round(image_p.y())));

  // really need these?
  if (image_p_i.x() < roi_min.x()) {
    image_p_i.x() = roi_min.x();
  }
  if (image_p_i.y() < roi_min.y()) {
    image_p_i.y() = roi_min.y();
  }
  if (roi_max.x() < image_p_i.x()) {
    image_p_i.x() = roi_max.x();
  }
  if (roi_max.y() < image_p_i.y()) {
    image_p_i.y() = roi_max.y();
  }

  return sdf.at(image_p_i.x(), image_p_i.y(), 0);
}

inline float SdfInterpolationBiliner(const Eigen::Vector2f& image_p,
                                     const vacancy::Image1f& sdf,
                                     const Eigen::Vector2i& roi_min,
                                     const Eigen::Vector2i& roi_max) {
  std::array<int, 2> pos_min = {{0, 0}};
  std::array<int, 2> pos_max = {{0, 0}};
  pos_min[0] = static_cast<int>(std::floor(image_p[0]));
  pos_min[1] = static_cast<int>(std::floor(image_p[1]));
  pos_max[0] = pos_min[0] + 1;
  pos_max[1] = pos_min[1] + 1;

  // really need these?
  if (pos_min[0] < roi_min.x()) {
    pos_min[0] = roi_min.x();
  }
  if (pos_min[1] < roi_min.y()) {
    pos_min[1] = roi_min.y();
  }
  if (roi_max.x() < pos_max[0]) {
    pos_max[0] = roi_max.x();
  }
  if (roi_max.y() < pos_max[1]) {
    pos_max[1] = roi_max.y();
  }

  float local_u = image_p[0] - pos_min[0];
  float local_v = image_p[1] - pos_min[1];

  // bilinear interpolation of sdf
  float dist =
      (1.0f - local_u) * (1.0f - local_v) * sdf.at(pos_min[0], pos_min[1], 0) +
      local_u * (1.0f - local_v) * sdf.at(pos_max[0], pos_min[1], 0) +
      (1.0f - local_u) * local_v * sdf.at(pos_min[0], pos_max[1], 0) +
      local_u * local_v * sdf.at(pos_max[0], pos_max[1], 0);

  return dist;
}

inline void UpdateVoxelMax(vacancy::Voxel* voxel,
                           const vacancy::VoxelUpdateOption& option,
                           float sdf) {
  (void)option;
  if (sdf > voxel->sdf) {
    voxel->sdf = sdf;
    voxel->update_num++;
  }
}

inline void UpdateVoxelWeightedAverage(vacancy::Voxel* voxel,
                                       const vacancy::VoxelUpdateOption& option,
                                       float sdf) {
  const float& w = option.voxel_update_weight;
  const float inv_denom = 1.0f / (w * (voxel->update_num + 1));
  voxel->sdf = (w * voxel->update_num * voxel->sdf + w * sdf) * inv_denom;
  voxel->update_num++;
}
}  // namespace

namespace vacancy {

const float InvalidSdf::kVal = std::numeric_limits<float>::lowest();

void DistanceTransformL1(const Image1b& mask, const Eigen::Vector2i& roi_min,
                         const Eigen::Vector2i& roi_max, Image1f* dist) {
  dist->Init(mask.width(), mask.height(), 0.0f);

  // init inifinite inside mask
  for (int y = roi_min.y(); y <= roi_max.y(); y++) {
    for (int x = roi_min.x(); x <= roi_max.x(); x++) {
      if (mask.at(x, y, 0) != 255) {
        continue;
      }
      dist->at(x, y, 0) = std::numeric_limits<float>::max();
    }
  }

  // forward path
  for (int y = roi_min.y() + 1; y <= roi_max.y(); y++) {
    float up = dist->at(roi_min.x(), y - 1, 0);
    if (up < std::numeric_limits<float>::max()) {
      dist->at(roi_min.x(), y, 0) =
          std::min(up + 1.0f, dist->at(roi_min.x(), y, 0));
    }
  }
  for (int x = roi_min.x() + 1; x <= roi_max.x(); x++) {
    float left = dist->at(x - 1, roi_min.y(), 0);
    if (left < std::numeric_limits<float>::max()) {
      dist->at(x, roi_min.y(), 0) =
          std::min(left + 1.0f, dist->at(x, roi_min.y(), 0));
    }
  }
  for (int y = roi_min.y() + 1; y <= roi_max.y(); y++) {
    for (int x = roi_min.x() + 1; x <= roi_max.x(); x++) {
      float up = dist->at(x, y - 1, 0);
      float left = dist->at(x - 1, y, 0);
      float min_dist = std::min(up, left);
      if (min_dist < std::numeric_limits<float>::max()) {
        dist->at(x, y, 0) = std::min(min_dist + 1.0f, dist->at(x, y, 0));
      }
    }
  }

  // backward path
  for (int y = roi_max.y() - 1; roi_min.y() <= y; y--) {
    float down = dist->at(roi_max.x(), y + 1, 0);
    if (down < std::numeric_limits<float>::max()) {
      dist->at(roi_max.x(), y, 0) =
          std::min(down + 1.0f, dist->at(roi_max.x(), y, 0));
    }
  }
  for (int x = roi_max.x() - 1; roi_min.x() <= x; x--) {
    float right = dist->at(x + 1, roi_max.y(), 0);
    if (right < std::numeric_limits<float>::max()) {
      dist->at(x, roi_max.y(), 0) =
          std::min(right + 1.0f, dist->at(x, roi_max.y(), 0));
    }
  }
  for (int y = roi_max.y() - 1; roi_min.y() <= y; y--) {
    for (int x = roi_max.x() - 1; roi_min.x() <= x; x--) {
      float down = dist->at(x, y + 1, 0);
      float right = dist->at(x + 1, y, 0);
      float min_dist = std::min(down, right);
      if (min_dist < std::numeric_limits<float>::max()) {
        dist->at(x, y, 0) = std::min(min_dist + 1.0f, dist->at(x, y, 0));
      }
    }
  }
}

void MakeSignedDistanceField(const Image1b& mask,
                             const Eigen::Vector2i& roi_min,
                             const Eigen::Vector2i& roi_max, Image1f* dist,
                             bool minmax_normalize, bool use_truncation,
                             float truncation_band) {
  Image1f* negative_dist = dist;
  DistanceTransformL1(mask, roi_min, roi_max, negative_dist);
  for (int y = roi_min.y(); y <= roi_max.y(); y++) {
    for (int x = roi_min.x(); x <= roi_max.x(); x++) {
      if (negative_dist->at(x, y, 0) > 0) {
        negative_dist->at(x, y, 0) *= -1;
      }
    }
  }

  Image1b inv_mask(mask);
  for (int y = roi_min.y(); y <= roi_max.y(); y++) {
    for (int x = roi_min.x(); x <= roi_max.x(); x++) {
      if (inv_mask.at(x, y, 0) == 255) {
        inv_mask.at(x, y, 0) = 0;
      } else {
        inv_mask.at(x, y, 0) = 255;
      }
    }
  }

  Image1f positive_dist;
  DistanceTransformL1(inv_mask, roi_min, roi_max, &positive_dist);
  for (int y = roi_min.y(); y <= roi_max.y(); y++) {
    for (int x = roi_min.x(); x <= roi_max.x(); x++) {
      if (inv_mask.at(x, y, 0) == 255) {
        dist->at(x, y, 0) = positive_dist.at(x, y, 0);
      }
    }
  }

  if (minmax_normalize) {
    // Outside of roi is set to 0, so does not affect min/max
    float max_dist =
        *std::max_element(dist->data().begin(), dist->data().end());
    float min_dist =
        *std::min_element(dist->data().begin(), dist->data().end());
    float abs_max = std::max(std::abs(max_dist), std::abs(min_dist));

    if (abs_max > std::numeric_limits<float>::min()) {
      float norm_factor = 1.0f / abs_max;

      for (int y = roi_min.y(); y <= roi_max.y(); y++) {
        for (int x = roi_min.x(); x <= roi_max.x(); x++) {
          dist->at(x, y, 0) *= norm_factor;
        }
      }
    }
  }

  // truncation process same to KinectFusion
  if (use_truncation) {
    for (int y = roi_min.y(); y <= roi_max.y(); y++) {
      for (int x = roi_min.x(); x <= roi_max.x(); x++) {
        float& d = dist->at(x, y, 0);
        if (-truncation_band >= d) {
          d = InvalidSdf::kVal;
        } else {
          d = std::min(1.0f, d / truncation_band);
        }
      }
    }
  }
}

void SignedDistance2Color(const Image1f& sdf, Image3b* vis_sdf,
                          float min_negative_d, float max_positive_d) {
  assert(min_negative_d < 0);
  assert(0 < max_positive_d);
  assert(vis_sdf != nullptr);

  vis_sdf->Init(sdf.width(), sdf.height());

  for (int y = 0; y < vis_sdf->height(); y++) {
    for (int x = 0; x < vis_sdf->width(); x++) {
      auto d = sdf.at(x, y, 0);

      if (d > 0) {
        float norm_inv_dist = (max_positive_d - d) / max_positive_d;
        norm_inv_dist = std::min(std::max(norm_inv_dist, 0.0f), 1.0f);
        vis_sdf->at(x, y, 0) = static_cast<uint8_t>(255);
        vis_sdf->at(x, y, 1) = static_cast<uint8_t>(255 * norm_inv_dist);
        vis_sdf->at(x, y, 2) = static_cast<uint8_t>(255 * norm_inv_dist);

      } else {
        float norm_inv_dist = (d - min_negative_d) / (-min_negative_d);
        norm_inv_dist = std::min(std::max(norm_inv_dist, 0.0f), 1.0f);
        vis_sdf->at(x, y, 0) = static_cast<uint8_t>(255 * norm_inv_dist);
        vis_sdf->at(x, y, 1) = static_cast<uint8_t>(255 * norm_inv_dist);
        vis_sdf->at(x, y, 2) = static_cast<uint8_t>(255);
      }
    }
  }
}

Voxel::Voxel() {}
Voxel::~Voxel() {}

VoxelGrid::VoxelGrid() {}

VoxelGrid::~VoxelGrid() {}

bool VoxelGrid::Init(const Eigen::Vector3f& bb_max,
                     const Eigen::Vector3f& bb_min, float resolution) {
  if (resolution < std::numeric_limits<float>::min()) {
    LOGE("resolution must be positive %f\n", resolution);
    return false;
  }
  if (bb_max.x() <= bb_min.x() || bb_max.y() <= bb_min.y() ||
      bb_max.z() <= bb_min.z()) {
    LOGE("input bounding box is invalid\n");
    return false;
  }

  bb_max_ = bb_max;
  bb_min_ = bb_min;
  resolution_ = resolution;

  Eigen::Vector3f diff = bb_max_ - bb_min_;

  for (int i = 0; i < 3; i++) {
    voxel_num_[i] = static_cast<int>(diff[i] / resolution_);
  }

  if (voxel_num_.x() * voxel_num_.y() * voxel_num_.z() >
      std::numeric_limits<int>::max()) {
    LOGE("too many voxels\n");
    return false;
  }

  xy_slice_num_ = voxel_num_[0] * voxel_num_[1];

  voxels_.clear();
  voxels_.resize(voxel_num_.x() * voxel_num_.y() * voxel_num_.z());

  float offset = resolution_ * 0.5f;

#if defined(_OPENMP) && defined(VACANCY_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int z = 0; z < voxel_num_.z(); z++) {
    float z_pos = diff.z() * (static_cast<float>(z) /
                              static_cast<float>(voxel_num_.z())) +
                  bb_min_.z() + offset;

    for (int y = 0; y < voxel_num_.y(); y++) {
      float y_pos = diff.y() * (static_cast<float>(y) /
                                static_cast<float>(voxel_num_.y())) +
                    bb_min_.y() + offset;
      for (int x = 0; x < voxel_num_.x(); x++) {
        float x_pos = diff.x() * (static_cast<float>(x) /
                                  static_cast<float>(voxel_num_.x())) +
                      bb_min_.x() + offset;

        Voxel* voxel = get_ptr(x, y, z);
        voxel->index.x() = x;
        voxel->index.y() = y;
        voxel->index.z() = z;

        voxel->id = z * xy_slice_num_ + (y * voxel_num_.x() + x);

        voxel->pos.x() = x_pos;
        voxel->pos.y() = y_pos;
        voxel->pos.z() = z_pos;

        voxel->sdf = InvalidSdf::kVal;
      }
    }
  }

  return true;
}

const Eigen::Vector3i& VoxelGrid::voxel_num() const { return voxel_num_; }

const Voxel& VoxelGrid::get(int x, int y, int z) const {
  return voxels_[z * xy_slice_num_ + (y * voxel_num_.x() + x)];
}

Voxel* VoxelGrid::get_ptr(int x, int y, int z) {
  return &voxels_[z * xy_slice_num_ + (y * voxel_num_.x() + x)];
}

float VoxelGrid::resolution() const { return resolution_; }

void VoxelGrid::ResetOnSurface() {
  for (Voxel& v : voxels_) {
    v.on_surface = false;
  }
}

bool VoxelGrid::initialized() const { return !voxels_.empty(); }

VoxelCarver::VoxelCarver() {}

VoxelCarver::~VoxelCarver() {}

VoxelCarver::VoxelCarver(VoxelCarverOption option) { set_option(option); }

void VoxelCarver::set_option(VoxelCarverOption option) { option_ = option; }

bool VoxelCarver::Init() {
  if (option_.update_option.voxel_max_update_num < 1) {
    LOGE("voxel_max_update_num must be positive");
    return false;
  }
  if (option_.update_option.voxel_update_weight <
      std::numeric_limits<float>::min()) {
    LOGE("voxel_update_weight must be positive");
    return false;
  }
  if (option_.update_option.truncation_band <
      std::numeric_limits<float>::min()) {
    LOGE("truncation_band must be positive");
    return false;
  }
  voxel_grid_ = std::make_unique<VoxelGrid>();
  return voxel_grid_->Init(option_.bb_max, option_.bb_min, option_.resolution);
}

bool VoxelCarver::Carve(const Camera& camera, const Image1b& silhouette,
                        const Eigen::Vector2i& roi_min,
                        const Eigen::Vector2i& roi_max, Image1f* sdf) {
  if (!voxel_grid_->initialized()) {
    LOGE("VoxelCarver::Carve voxel grid has not been initialized\n");
    return false;
  }

  Timer<> timer;
  timer.Start();
  // make signed distance field
  MakeSignedDistanceField(silhouette, roi_min, roi_max, sdf,
                          option_.sdf_minmax_normalize,
                          option_.update_option.use_truncation,
                          option_.update_option.truncation_band);
  timer.End();
  LOGI("VoxelCarver::Carve make SDF %02f\n", timer.elapsed_msec());

  return Carve(camera, roi_min, roi_max, *sdf);
}

bool VoxelCarver::Carve(const Camera& camera, const Eigen::Vector2i& roi_min,
                        const Eigen::Vector2i& roi_max, const Image1f& sdf) {
  Timer<> timer;
  std::function<float(const Eigen::Vector2f&, const vacancy::Image1f&,
                      const Eigen::Vector2i&, const Eigen::Vector2i&)>
      interpolate_sdf;
  if (option_.update_option.sdf_interp == SdfInterpolation::kNn) {
    interpolate_sdf = SdfInterpolationNn;
  } else if (option_.update_option.sdf_interp == SdfInterpolation::kBilinear) {
    interpolate_sdf = SdfInterpolationBiliner;
  }

  std::function<void(Voxel*, const VoxelUpdateOption&, float)> update_voxel;
  if (option_.update_option.voxel_update == VoxelUpdate::kMax) {
    update_voxel = UpdateVoxelMax;
  } else if (option_.update_option.voxel_update ==
             VoxelUpdate::kWeightedAverage) {
    update_voxel = UpdateVoxelWeightedAverage;
  }

  timer.Start();
  const float max_sdf = *std::max_element(sdf.data().begin(), sdf.data().end());
  const Eigen::Vector3i& voxel_num = voxel_grid_->voxel_num();
  const Eigen::Affine3f& w2c = camera.w2c().cast<float>();
#if defined(_OPENMP) && defined(VACANCY_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int z = 0; z < voxel_num.z(); z++) {
    for (int y = 0; y < voxel_num.y(); y++) {
      for (int x = 0; x < voxel_num.x(); x++) {
        Voxel* voxel = voxel_grid_->get_ptr(x, y, z);

        if (voxel->outside ||
            voxel->update_num > option_.update_option.voxel_max_update_num) {
          continue;
        }

        Eigen::Vector2f image_p_f;
        Eigen::Vector3f voxel_pos_c = w2c * voxel->pos;

        // skip if the voxel is in the back of the camera
        if (voxel_pos_c.z() < 0) {
          continue;
        }

        camera.Project(voxel_pos_c, &image_p_f);

        float dist = InvalidSdf::kVal;

        if (image_p_f.x() < roi_min.x() || image_p_f.y() < roi_min.y() ||
            roi_max.x() < image_p_f.x() || roi_max.y() < image_p_f.y()) {
          if (option_.update_option.update_outside ==
              UpdateOutsideImage::kNone) {
            continue;
          } else if (option_.update_option.update_outside ==
                     UpdateOutsideImage::kMax) {
            dist = max_sdf;
          }
        } else {
          dist = interpolate_sdf(image_p_f, sdf, roi_min, roi_max);
        }

        // skip if dist is truncated
        if (option_.update_option.use_truncation && dist < -1.0f) {
          continue;
        }

        if (voxel->update_num < 1) {
          voxel->sdf = dist;
          voxel->update_num++;
          continue;
        }

        update_voxel(voxel, option_.update_option, dist);
      }
    }
  }
  timer.End();
  LOGI("VoxelCarver::Carve main loop %02f\n", timer.elapsed_msec());

  return true;
}

bool VoxelCarver::Carve(const Camera& camera, const Image1b& silhouette) {
  Image1f sdf(camera.width(), camera.height());
  return Carve(camera, silhouette, &sdf);
}

bool VoxelCarver::Carve(const Camera& camera, const Image1b& silhouette,
                        Image1f* sdf) {
  Eigen::Vector2i roi_min{0, 0};
  Eigen::Vector2i roi_max{silhouette.width() - 1, silhouette.height() - 1};
  return Carve(camera, silhouette, roi_min, roi_max, sdf);
}

bool VoxelCarver::Carve(const Camera& camera, const Image1f& sdf) {
  Eigen::Vector2i roi_min{0, 0};
  Eigen::Vector2i roi_max{sdf.width() - 1, sdf.height() - 1};
  return Carve(camera, roi_min, roi_max, sdf);
}

bool VoxelCarver::Carve(const std::vector<Camera>& cameras,
                        const std::vector<Image1b>& silhouettes) {
  assert(cameras.size() == silhouettes.size());

  for (size_t i = 0; i < cameras.size(); i++) {
    bool ret = Carve(cameras[i], silhouettes[i]);
    if (!ret) {
      return false;
    }
  }

  return true;
}

void VoxelCarver::ExtractVoxel(Mesh* mesh, bool inside_empty) {
  Timer<> timer;
  timer.Start();

  vacancy::ExtractVoxel(voxel_grid_.get(), mesh, inside_empty);

  timer.End();
  LOGI("VoxelCarver::ExtractVoxel %02f\n", timer.elapsed_msec());
}

void VoxelCarver::ExtractIsoSurface(Mesh* mesh, double iso_level,
                                    bool linear_interp) {
  MarchingCubes(*voxel_grid_, mesh, iso_level, linear_interp);
}

}  // namespace vacancy
