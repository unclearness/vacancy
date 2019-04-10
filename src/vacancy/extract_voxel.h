/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include "vacancy/voxel_carver.h"

namespace vacancy {

void ExtractVoxel(VoxelGrid* voxel_grid, float resolution, Mesh* mesh,
                  bool inside_empty, bool with_pseudo_surface);

}  // namespace vacancy
