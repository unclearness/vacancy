/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

// original code is from
// http://paulbourke.net/geometry/polygonise/

#pragma once

#include "vacancy/voxel_carver.h"

namespace vacancy {

void MarchingCubes(const VoxelGrid& voxel_grid, Mesh* mesh,
                   double iso_level = 0.0, bool linear_interp = true);

}  // namespace vacancy
