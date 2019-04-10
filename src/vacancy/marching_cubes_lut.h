/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

// original code is from
// http://paulbourke.net/geometry/polygonise/

#pragma once

#include <array>

namespace vacancy {
namespace marching_cubes_lut {
extern std::array<int, 256> kEdgeTable;
extern std::array<std::array<int, 16>, 256> kTriTable;
}  // namespace marching_cubes_lut
}  // namespace vacancy
