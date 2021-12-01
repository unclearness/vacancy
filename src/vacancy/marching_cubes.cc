/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

// original code is from
// http://paulbourke.net/geometry/polygonise/

#include "vacancy/marching_cubes.h"

#include <array>
#include <map>
#include <utility>
#include <vector>

#include "vacancy/marching_cubes_lut.h"
#include "vacancy/timer.h"

namespace {

/*
   Linearly interpolate the position where an isosurface cuts
   an edge between two vertices, each with their own scalar value
*/
void VertexInterp(double isolevel, const Eigen::Vector3f &p1,
                  const Eigen::Vector3f &p2, double valp1, double valp2,
                  Eigen::Vector3f *p) {
  if (std::abs(isolevel - valp1) < 0.00001) {
    *p = p1;
    return;
  }
  if (std::abs(isolevel - valp2) < 0.00001) {
    *p = p2;
    return;
  }
  if (std::abs(valp1 - valp2) < 0.00001) {
    *p = p1;
    return;
  }
  double mu = (isolevel - valp1) / (valp2 - valp1);
  p->x() = static_cast<float>(p1.x() + mu * (static_cast<double>(p2.x()) -
                                             static_cast<double>(p1.x())));
  p->y() = static_cast<float>(p1.y() + mu * (static_cast<double>(p2.y()) -
                                             static_cast<double>(p1.y())));
  p->z() = static_cast<float>(p1.z() + mu * (static_cast<double>(p2.z()) -
                                             static_cast<double>(p1.z())));
}

void VertexInterp(double iso_level, const vacancy::Voxel &v1,
                  const vacancy::Voxel &v2, Eigen::Vector3f *p,
                  bool linear_interp) {
  if (linear_interp) {
    VertexInterp(iso_level, v1.pos, v2.pos, v1.sdf, v2.sdf, p);
  } else {
    *p = v1.pos;
  }
}

}  // namespace

namespace vacancy {

void MarchingCubes(const VoxelGrid &voxel_grid, Mesh *mesh, double iso_level,
                   bool linear_interp) {
  Timer<> timer;
  timer.Start();

  mesh->Clear();

  const Eigen::Vector3i &voxel_num = voxel_grid.voxel_num();

  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3i> vertex_indices;

  // key: sorted a pair of unique voxel ids in the voxel grid
  // value: the corresponding interpolated vertex id in the unified mesh
  // todo: consider faster way... maybe unordered_map?
  std::map<std::pair<int, int>, int> voxelids2vertexid;

  const std::array<int, 256> &edge_table =
      vacancy::marching_cubes_lut::kEdgeTable;
  const std::array<std::array<int, 16>, 256> &tri_table =
      vacancy::marching_cubes_lut::kTriTable;

  for (int z = 1; z < voxel_num.z(); z++) {
    for (int y = 1; y < voxel_num.y(); y++) {
      for (int x = 1; x < voxel_num.x(); x++) {
        if (voxel_grid.get(x, y, z).update_num < 1) {
          continue;
        }

        std::array<const Voxel *, 8> voxels;
        voxels[0] = &voxel_grid.get(x - 1, y - 1, z - 1);
        voxels[1] = &voxel_grid.get(x, y - 1, z - 1);
        voxels[2] = &voxel_grid.get(x, y, z - 1);
        voxels[3] = &voxel_grid.get(x - 1, y, z - 1);

        voxels[4] = &voxel_grid.get(x - 1, y - 1, z);
        voxels[5] = &voxel_grid.get(x, y - 1, z);
        voxels[6] = &voxel_grid.get(x, y, z);
        voxels[7] = &voxel_grid.get(x - 1, y, z);

        if (voxels[0]->sdf == InvalidSdf::kVal ||
            voxels[1]->sdf == InvalidSdf::kVal ||
            voxels[2]->sdf == InvalidSdf::kVal ||
            voxels[3]->sdf == InvalidSdf::kVal ||
            voxels[4]->sdf == InvalidSdf::kVal ||
            voxels[5]->sdf == InvalidSdf::kVal ||
            voxels[6]->sdf == InvalidSdf::kVal ||
            voxels[7]->sdf == InvalidSdf::kVal) {
          continue;
        }

        int cube_index{0};
        std::array<Eigen::Vector3f, 12> vert_list;
        std::array<std::pair<int, int>, 12> voxelids_list;
        /*
           Determine the index into the edge table which
           tells us which vertices are inside of the surface
        */
        if (voxels[0]->sdf < iso_level) cube_index |= 1;
        if (voxels[1]->sdf < iso_level) cube_index |= 2;
        if (voxels[2]->sdf < iso_level) cube_index |= 4;
        if (voxels[3]->sdf < iso_level) cube_index |= 8;
        if (voxels[4]->sdf < iso_level) cube_index |= 16;
        if (voxels[5]->sdf < iso_level) cube_index |= 32;
        if (voxels[6]->sdf < iso_level) cube_index |= 64;
        if (voxels[7]->sdf < iso_level) cube_index |= 128;

        /* Cube is entirely in/out of the surface */
        if (edge_table[cube_index] == 0) {
          continue;
        }

        /* Find the vertices where the surface intersects the cube
         * And save a pair of voxel ids when the vertices occur
         */
        if (edge_table[cube_index] & 1) {
          VertexInterp(iso_level, *voxels[0], *voxels[1], &vert_list[0],
                       linear_interp);
          voxelids_list[0] = std::make_pair(voxels[0]->id, voxels[1]->id);
        }
        if (edge_table[cube_index] & 2) {
          VertexInterp(iso_level, *voxels[1], *voxels[2], &vert_list[1],
                       linear_interp);
          voxelids_list[1] = std::make_pair(voxels[1]->id, voxels[2]->id);
        }
        if (edge_table[cube_index] & 4) {
          VertexInterp(iso_level, *voxels[2], *voxels[3], &vert_list[2],
                       linear_interp);
          voxelids_list[2] = std::make_pair(voxels[3]->id, voxels[2]->id);
        }
        if (edge_table[cube_index] & 8) {
          VertexInterp(iso_level, *voxels[3], *voxels[0], &vert_list[3],
                       linear_interp);
          voxelids_list[3] = std::make_pair(voxels[0]->id, voxels[3]->id);
        }
        if (edge_table[cube_index] & 16) {
          VertexInterp(iso_level, *voxels[4], *voxels[5], &vert_list[4],
                       linear_interp);
          voxelids_list[4] = std::make_pair(voxels[4]->id, voxels[5]->id);
        }
        if (edge_table[cube_index] & 32) {
          VertexInterp(iso_level, *voxels[5], *voxels[6], &vert_list[5],
                       linear_interp);
          voxelids_list[5] = std::make_pair(voxels[5]->id, voxels[6]->id);
        }
        if (edge_table[cube_index] & 64) {
          VertexInterp(iso_level, *voxels[6], *voxels[7], &vert_list[6],
                       linear_interp);
          voxelids_list[6] = std::make_pair(voxels[7]->id, voxels[6]->id);
        }
        if (edge_table[cube_index] & 128) {
          VertexInterp(iso_level, *voxels[7], *voxels[4], &vert_list[7],
                       linear_interp);
          voxelids_list[7] = std::make_pair(voxels[4]->id, voxels[7]->id);
        }
        if (edge_table[cube_index] & 256) {
          VertexInterp(iso_level, *voxels[0], *voxels[4], &vert_list[8],
                       linear_interp);
          voxelids_list[8] = std::make_pair(voxels[0]->id, voxels[4]->id);
        }
        if (edge_table[cube_index] & 512) {
          VertexInterp(iso_level, *voxels[1], *voxels[5], &vert_list[9],
                       linear_interp);
          voxelids_list[9] = std::make_pair(voxels[1]->id, voxels[5]->id);
        }
        if (edge_table[cube_index] & 1024) {
          VertexInterp(iso_level, *voxels[2], *voxels[6], &vert_list[10],
                       linear_interp);
          voxelids_list[10] = std::make_pair(voxels[2]->id, voxels[6]->id);
        }
        if (edge_table[cube_index] & 2048) {
          VertexInterp(iso_level, *voxels[3], *voxels[7], &vert_list[11],
                       linear_interp);
          voxelids_list[11] = std::make_pair(voxels[3]->id, voxels[7]->id);
        }

        for (int i = 0; tri_table[cube_index][i] != -1; i += 3) {
          Eigen::Vector3i face;

          for (int j = 0; j < 3; j++) {
            const std::pair<int, int> &key =
                voxelids_list[tri_table[cube_index][i + (2 - j)]];
            if (voxelids2vertexid.find(key) == voxelids2vertexid.end()) {
              // if a pair of voxel ids has not been added, the current vertex
              // is new. store the pair of voxel ids and new vertex position
              face[j] = static_cast<int>(vertices.size());
              vertices.push_back(vert_list[tri_table[cube_index][i + (2 - j)]]);
              voxelids2vertexid.insert(std::make_pair(key, face[j]));
            } else {
              // set existing vertex id if the pair of voxel ids has been
              // already added
              face[j] = voxelids2vertexid.at(key);
            }
          }
          vertex_indices.push_back(face);
        }
      }
    }
  }

  mesh->set_vertices(vertices);
  mesh->set_vertex_indices(vertex_indices);

  timer.End();
  LOGI("MarchingCubes %02f\n", timer.elapsed_msec());
}

}  // namespace vacancy
