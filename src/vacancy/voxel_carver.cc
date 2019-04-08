/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "vacancy/voxel_carver.h"
#include "vacancy/marching_cubes.h"
#include "vacancy/timer.h"

namespace vacancy {

void DistanceTransformL1(const Image1b& mask, Image1f* dist) {
  dist->Init(mask.width(), mask.height(), 0.0f);

  // init inifinite inside mask
  for (int y = 0; y < mask.height(); y++) {
    for (int x = 0; x < mask.width(); x++) {
      if (mask.at(x, y, 0) != 255) {
        continue;
      }
      dist->at(x, y, 0) = std::numeric_limits<float>::max();
    }
  }

  // forward path
  for (int y = 1; y < mask.height(); y++) {
    float up = dist->at(0, y - 1, 0);
    if (up < std::numeric_limits<float>::max()) {
      dist->at(0, y, 0) = std::min(up + 1.0f, dist->at(0, y, 0));
    }
  }
  for (int x = 1; x < mask.width(); x++) {
    float left = dist->at(x - 1, 0, 0);
    if (left < std::numeric_limits<float>::max()) {
      dist->at(x, 0, 0) = std::min(left + 1.0f, dist->at(x, 0, 0));
    }
  }
  for (int y = 1; y < mask.height(); y++) {
    for (int x = 1; x < mask.width(); x++) {
      float up = dist->at(x, y - 1, 0);
      float left = dist->at(x - 1, y, 0);
      float min_dist = std::min(up, left);
      if (min_dist < std::numeric_limits<float>::max()) {
        dist->at(x, y, 0) = std::min(min_dist + 1.0f, dist->at(x, y, 0));
      }
    }
  }

  // backward path
  for (int y = mask.height() - 2; 0 <= y; y--) {
    float down = dist->at(mask.width() - 1, y + 1, 0);
    if (down < std::numeric_limits<float>::max()) {
      dist->at(mask.width() - 1, y, 0) =
          std::min(down + 1.0f, dist->at(mask.width() - 1, y, 0));
    }
  }
  for (int x = mask.width() - 2; 0 <= x; x--) {
    float right = dist->at(x + 1, mask.height() - 1, 0);
    if (right < std::numeric_limits<float>::max()) {
      dist->at(x, mask.height() - 1, 0) =
          std::min(right + 1.0f, dist->at(x, mask.height() - 1, 0));
    }
  }
  for (int y = mask.height() - 2; 0 <= y; y--) {
    for (int x = mask.width() - 2; 0 <= x; x--) {
      float down = dist->at(x, y + 1, 0);
      float right = dist->at(x + 1, y, 0);
      float min_dist = std::min(down, right);
      if (min_dist < std::numeric_limits<float>::max()) {
        dist->at(x, y, 0) = std::min(min_dist + 1.0f, dist->at(x, y, 0));
      }
    }
  }
}

void MakeSignedDistanceField(const Image1b& mask, Image1f* dist,
                             bool minmax_normalize, bool use_truncation,
                             float truncation_band) {
  Image1f* negative_dist = dist;
  DistanceTransformL1(mask, negative_dist);
  for (int y = 0; y < negative_dist->height(); y++) {
    for (int x = 0; x < negative_dist->width(); x++) {
      if (negative_dist->at(x, y, 0) > 0) {
        negative_dist->at(x, y, 0) *= -1;
      }
    }
  }

  Image1b inv_mask(mask);
  for (int y = 0; y < inv_mask.height(); y++) {
    for (int x = 0; x < inv_mask.width(); x++) {
      if (inv_mask.at(x, y, 0) == 255) {
        inv_mask.at(x, y, 0) = 0;
      } else {
        inv_mask.at(x, y, 0) = 255;
      }
    }
  }

  Image1f positive_dist;
  DistanceTransformL1(inv_mask, &positive_dist);
  for (int y = 0; y < inv_mask.height(); y++) {
    for (int x = 0; x < inv_mask.width(); x++) {
      if (inv_mask.at(x, y, 0) == 255) {
        dist->at(x, y, 0) = positive_dist.at(x, y, 0);
      }
    }
  }

  if (minmax_normalize) {
    float max_dist =
        *std::max_element(dist->data().begin(), dist->data().end());
    float min_dist =
        *std::min_element(dist->data().begin(), dist->data().end());
    float abs_max = std::max(std::abs(max_dist), std::abs(min_dist));

    if (abs_max > std::numeric_limits<float>::min()) {
      float norm_factor = 1.0f / abs_max;

      for (int y = 0; y < dist->height(); y++) {
        for (int x = 0; x < dist->width(); x++) {
          dist->at(x, y, 0) *= norm_factor;
        }
      }
    }
  }

  // truncation process same to KinectFusion
  if (use_truncation) {
    for (int y = 0; y < dist->height(); y++) {
      for (int x = 0; x < dist->width(); x++) {
        float& d = dist->at(x, y, 0);
        if (-truncation_band >= d) {
          d = std::numeric_limits<float>::lowest();
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

Voxel::Voxel(){}
Voxel::~Voxel() {}

VoxelGrid::VoxelGrid() {}

VoxelGrid::~VoxelGrid() {}

bool VoxelGrid::Init(const Eigen::Vector3f& bb_max,
                     const Eigen::Vector3f& bb_min, float resolution) {
  if (resolution < 0) {
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

  xy_slice_num_ = voxel_num_[0] * voxel_num_[1];

  voxels_.clear();
  voxels_.resize(voxel_num_.x() * voxel_num_.y() * voxel_num_.z());

  float offset = resolution_ * 0.5f;

  const float min_dist =
      std::numeric_limits<float>::lowest();  // std::max(std::max(diff[0],
                                             // diff[1]), diff[2]);

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

        voxel->pos.x() = x_pos;
        voxel->pos.y() = y_pos;
        voxel->pos.z() = z_pos;

        voxel->sdf = min_dist;
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
  voxel_grid_ = std::make_unique<VoxelGrid>();
  return voxel_grid_->Init(option_.bb_max, option_.bb_min, option_.resolution);
}

bool VoxelCarver::Carve(const Camera& camera, const Image1b& silhouette,
                        Image1f* sdf) {
  if (!voxel_grid_->initialized()) {
    LOGE("VoxelCarver::Carve voxel grid has not been initialized\n");
    return false;
  }

  Timer<> timer;
  timer.Start();
  // make signed distance field
  MakeSignedDistanceField(silhouette, sdf, option_.sdf_minmax_normalize,
                          option_.update_option.use_truncation,
                          option_.update_option.truncation_band);
  timer.End();
  LOGI("VoxelCarver::Carve make SDF %02f\n", timer.elapsed_msec());

  timer.Start();
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

        Eigen::Vector2i image_p(static_cast<int>(std::round(image_p_f.x())),
                                static_cast<int>(std::round(image_p_f.y())));
        if (image_p.x() < 0 || image_p.y() < 0 ||
            camera.width() - 1 < image_p.x() ||
            camera.height() - 1 < image_p.y()) {
          continue;
        }

        float dist = sdf->at(image_p.x(), image_p.y(), 0);

        // skip if dist is truncated
        if (option_.update_option.use_truncation && dist < -1.0f) {
          continue;
        }

        if (option_.update_option.voxel_update == VoxelUpdate::kMax) {
          if (dist > voxel->sdf) {
            voxel->sdf = dist;
            voxel->update_num++;
          }
        } else if (option_.update_option.voxel_update ==
                   VoxelUpdate::kWeightedAverage) {
          if (voxel->update_num < 1) {
            voxel->sdf = dist;
          } else {
            float w = option_.update_option.voxel_update_weight;
            float inv_denom = 1.0f / (w * (voxel->update_num + 1));
            voxel->sdf =
                (w * voxel->update_num * voxel->sdf + w * dist) * inv_denom;
          }
          voxel->update_num++;
        }
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

void VoxelCarver::UpdateOnSurface() {
  // raycast like surface detection
  // search xyz axes to detect the voxel on sign change

  const Eigen::Vector3i& voxel_num = voxel_grid_->voxel_num();

  constexpr float e = std::numeric_limits<float>::min();

  // x
  for (int z = 0; z < voxel_num.z(); z++) {
    for (int y = 0; y < voxel_num.y(); y++) {
      for (int x = 1; x < voxel_num.x(); x++) {
        const Voxel& prev_voxel = voxel_grid_->get(x - 1, y, z);
        Voxel* voxel = voxel_grid_->get_ptr(x, y, z);
        if (voxel->update_num < 1 || prev_voxel.update_num < 1) {
          continue;
        }
        if (voxel->sdf * prev_voxel.sdf < 0) {
          voxel->on_surface = true;
        }
        if (std::abs(voxel->sdf) < e) {
          voxel->on_surface = true;
        }
      }
    }
  }

  // y
  for (int z = 0; z < voxel_num.z(); z++) {
    for (int x = 0; x < voxel_num.x(); x++) {
      for (int y = 1; y < voxel_num.y(); y++) {
        const Voxel& prev_voxel = voxel_grid_->get(x, y - 1, z);
        Voxel* voxel = voxel_grid_->get_ptr(x, y, z);
        if (voxel->update_num < 1 || prev_voxel.update_num < 1) {
          continue;
        }
        if (voxel->sdf * prev_voxel.sdf < 0) {
          voxel->on_surface = true;
        }
        if (std::abs(voxel->sdf) < e) {
          voxel->on_surface = true;
        }
      }
    }
  }

  // z
  for (int y = 0; y < voxel_num.y(); y++) {
    for (int x = 0; x < voxel_num.x(); x++) {
      for (int z = 1; z < voxel_num.z(); z++) {
        const Voxel& prev_voxel = voxel_grid_->get(x, y, z - 1);
        Voxel* voxel = voxel_grid_->get_ptr(x, y, z);
        if (voxel->update_num < 1 || prev_voxel.update_num < 1) {
          continue;
        }
        if (voxel->sdf * prev_voxel.sdf < 0) {
          voxel->on_surface = true;
        }
        if (std::abs(voxel->sdf) < e) {
          voxel->on_surface = true;
        }
      }
    }
  }
}

void VoxelCarver::UpdateOnSurfaceWithPseudo() {
  // raycast like surface detection
  // search xyz axes bidirectionally to detect the voxel whose sign changes from
  // + to -
  // this function adds pseudo surfaces, which are good for visualization

  const Eigen::Vector3i& voxel_num = voxel_grid_->voxel_num();

  // x
  for (int z = 0; z < voxel_num.z(); z++) {
    for (int y = 0; y < voxel_num.y(); y++) {
      int min_index{-1};
      int max_index{-1};
      // min to max
      bool found_min{false};
      {
        float min_sdf = std::numeric_limits<float>::max();
        for (int x = 0; x < voxel_num.x(); x++) {
          const Voxel& voxel = voxel_grid_->get(x, y, z);

          if (voxel.update_num < 1) {
            continue;
          }

          if (voxel.sdf < min_sdf) {
            min_sdf = voxel.sdf;
            min_index = x;
          }
          if (min_sdf < 0) {
            found_min = true;
            break;
          }
        }
      }

      // max to min
      bool found_max{false};
      {
        float min_sdf = std::numeric_limits<float>::max();
        for (int x = voxel_num.x() - 1; 0 <= x; x--) {
          const Voxel& voxel = voxel_grid_->get(x, y, z);

          if (voxel.update_num < 1) {
            continue;
          }

          if (voxel.sdf < min_sdf) {
            min_sdf = voxel.sdf;
            max_index = x;
          }
          if (min_sdf < 0) {
            found_max = true;
            break;
          }
        }
      }

      if (found_min) {
        voxel_grid_->get_ptr(min_index, y, z)->on_surface = true;
      }
      if (found_max) {
        voxel_grid_->get_ptr(max_index, y, z)->on_surface = true;
      }
    }
  }

  // y
  for (int z = 0; z < voxel_num.z(); z++) {
    for (int x = 0; x < voxel_num.x(); x++) {
      int min_index{-1};
      int max_index{-1};
      // min to max
      bool found_min{false};
      {
        float min_sdf = std::numeric_limits<float>::max();
        for (int y = 0; y < voxel_num.y(); y++) {
          const Voxel& voxel = voxel_grid_->get(x, y, z);
          if (voxel.update_num < 1) {
            continue;
          }
          if (voxel.sdf < min_sdf) {
            min_sdf = voxel.sdf;
            min_index = y;
          }
          if (min_sdf < 0) {
            found_min = true;
            break;
          }
        }
      }

      // max to min
      bool found_max{false};
      {
        float min_sdf = std::numeric_limits<float>::max();
        for (int y = voxel_num.y() - 1; 0 <= y; y--) {
          const Voxel& voxel = voxel_grid_->get(x, y, z);
          if (voxel.update_num < 1) {
            continue;
          }
          if (voxel.sdf < min_sdf) {
            min_sdf = voxel.sdf;
            max_index = y;
          }
          if (min_sdf < 0) {
            found_max = true;
            break;
          }
        }
      }

      if (found_min) {
        voxel_grid_->get_ptr(x, min_index, z)->on_surface = true;
      }
      if (found_max) {
        voxel_grid_->get_ptr(x, max_index, z)->on_surface = true;
      }
    }
  }

  // z
  for (int y = 0; y < voxel_num.y(); y++) {
    for (int x = 0; x < voxel_num.x(); x++) {
      int min_index{-1};
      int max_index{-1};
      // min to max
      bool found_min{false};
      {
        float min_sdf = std::numeric_limits<float>::max();
        for (int z = 0; z < voxel_num.z(); z++) {
          const Voxel& voxel = voxel_grid_->get(x, y, z);
          if (voxel.update_num < 1) {
            continue;
          }
          if (voxel.sdf < min_sdf) {
            min_sdf = voxel.sdf;
            min_index = z;
          }
          if (min_sdf < 0) {
            found_min = true;
            break;
          }
        }
      }

      // max to min
      bool found_max{false};
      {
        float min_sdf = std::numeric_limits<float>::max();
        for (int z = voxel_num.z() - 1; 0 <= z; z--) {
          const Voxel& voxel = voxel_grid_->get(x, y, z);
          if (voxel.update_num < 1) {
            continue;
          }
          if (voxel.sdf < min_sdf) {
            min_sdf = voxel.sdf;
            max_index = z;
          }
          if (min_sdf < 0) {
            found_max = true;
            break;
          }
        }
      }

      if (found_min) {
        voxel_grid_->get_ptr(x, y, min_index)->on_surface = true;
      }
      if (found_max) {
        voxel_grid_->get_ptr(x, y, max_index)->on_surface = true;
      }
    }
  }
}

void VoxelCarver::ExtractVoxel(Mesh* mesh, bool inside_empty,
                               bool with_pseudo_surface) {
  const Eigen::Vector3i& voxel_num = voxel_grid_->voxel_num();

  Timer<> timer;
  timer.Start();

  mesh->Clear();

  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> vertex_indices;

  std::shared_ptr<Mesh> cube = MakeCube(option_.resolution);

  // update on_surface flag of voxels
  if (inside_empty) {
    voxel_grid_->ResetOnSurface();

    if (with_pseudo_surface) {
      UpdateOnSurfaceWithPseudo();
    } else {
      UpdateOnSurface();
    }
  }

  for (int z = 0; z < voxel_num.z(); z++) {
    for (int y = 0; y < voxel_num.y(); y++) {
      for (int x = 0; x < voxel_num.x(); x++) {
        const Voxel& voxel = voxel_grid_->get(x, y, z);

        if (inside_empty) {
          if (!voxel.on_surface) {
            // if inside_empty is specified, skip non-surface voxels
            continue;
          }
        } else if (voxel.sdf > 0 || voxel.update_num < 1 || voxel.outside) {
          // otherwise, naively skip outside and non-updated voxels
          continue;
        }

        cube->Translate(voxel.pos);

        int vertex_index_offset = static_cast<int>(vertices.size());

        std::copy(cube->vertices().begin(), cube->vertices().end(),
                  std::back_inserter(vertices));

        std::vector<Eigen::Vector3i> fixed_vertex_indices =
            cube->vertex_indices();

        std::for_each(fixed_vertex_indices.begin(), fixed_vertex_indices.end(),
                      [&](Eigen::Vector3i& face) {
                        face[0] += vertex_index_offset;
                        face[1] += vertex_index_offset;
                        face[2] += vertex_index_offset;
                      });

        std::copy(fixed_vertex_indices.begin(), fixed_vertex_indices.end(),
                  std::back_inserter(vertex_indices));

        cube->Translate(-voxel.pos);
      }
    }
  }

  mesh->set_vertices(vertices);
  mesh->set_vertex_indices(vertex_indices);

  timer.End();
  LOGI("VoxelCarver::ExtractVoxel %02f\n", timer.elapsed_msec());
}

void VoxelCarver::ExtractIsoSurface(Mesh* mesh, double iso_level) {
  MarchingCubes(*voxel_grid_, mesh, iso_level);
}

}  // namespace vacancy
