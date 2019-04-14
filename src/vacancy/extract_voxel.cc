/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "vacancy/extract_voxel.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

namespace {

void UpdateOnSurface(vacancy::VoxelGrid* voxel_grid) {
  // raycast like surface detection
  // search xyz axes to detect the voxel on sign change

  const Eigen::Vector3i& voxel_num = voxel_grid->voxel_num();

  constexpr float e = std::numeric_limits<float>::min();

  // x
  for (int z = 0; z < voxel_num.z(); z++) {
    for (int y = 0; y < voxel_num.y(); y++) {
      for (int x = 1; x < voxel_num.x(); x++) {
        const vacancy::Voxel& prev_voxel = voxel_grid->get(x - 1, y, z);
        vacancy::Voxel* voxel = voxel_grid->get_ptr(x, y, z);
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
        const vacancy::Voxel& prev_voxel = voxel_grid->get(x, y - 1, z);
        vacancy::Voxel* voxel = voxel_grid->get_ptr(x, y, z);
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
        const vacancy::Voxel& prev_voxel = voxel_grid->get(x, y, z - 1);
        vacancy::Voxel* voxel = voxel_grid->get_ptr(x, y, z);
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

void UpdateOnSurfaceWithPseudo(vacancy::VoxelGrid* voxel_grid) {
  // raycast like surface detection
  // search xyz axes bidirectionally to detect the voxel whose sign changes from
  // + to -
  // this function adds pseudo surfaces, which are good for visualization

  const Eigen::Vector3i& voxel_num = voxel_grid->voxel_num();

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
          const vacancy::Voxel& voxel = voxel_grid->get(x, y, z);

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
          const vacancy::Voxel& voxel = voxel_grid->get(x, y, z);

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
        voxel_grid->get_ptr(min_index, y, z)->on_surface = true;
      }
      if (found_max) {
        voxel_grid->get_ptr(max_index, y, z)->on_surface = true;
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
          const vacancy::Voxel& voxel = voxel_grid->get(x, y, z);
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
          const vacancy::Voxel& voxel = voxel_grid->get(x, y, z);
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
        voxel_grid->get_ptr(x, min_index, z)->on_surface = true;
      }
      if (found_max) {
        voxel_grid->get_ptr(x, max_index, z)->on_surface = true;
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
          const vacancy::Voxel& voxel = voxel_grid->get(x, y, z);
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
          const vacancy::Voxel& voxel = voxel_grid->get(x, y, z);
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
        voxel_grid->get_ptr(x, y, min_index)->on_surface = true;
      }
      if (found_max) {
        voxel_grid->get_ptr(x, y, max_index)->on_surface = true;
      }
    }
  }
}
}  // namespace

namespace vacancy {
void ExtractVoxel(VoxelGrid* voxel_grid, Mesh* mesh, bool inside_empty) {
  const Eigen::Vector3i& voxel_num = voxel_grid->voxel_num();

  mesh->Clear();

  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> vertex_indices;

  std::shared_ptr<Mesh> cube = MakeCube(voxel_grid->resolution());

  // update on_surface flag of voxels
  if (inside_empty) {
    voxel_grid->ResetOnSurface();

    UpdateOnSurface(voxel_grid);
  }

  for (int z = 0; z < voxel_num.z(); z++) {
    for (int y = 0; y < voxel_num.y(); y++) {
      for (int x = 0; x < voxel_num.x(); x++) {
        const Voxel& voxel = voxel_grid->get(x, y, z);

        if (inside_empty) {
          if (!voxel.on_surface) {
            // if inside_empty is specified, skip non-surface voxels
            continue;
          }
        } else if (voxel.sdf > 0 || voxel.update_num < 1) {
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
}

}  // namespace vacancy
