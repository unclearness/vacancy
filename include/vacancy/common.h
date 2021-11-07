/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <cassert>
#include <iomanip>
#include <limits>
#include <string>

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4127)
#pragma warning(disable : 4819)
#pragma warning(disable : 26451)
#pragma warning(disable : 26495)
#pragma warning(disable : 26812)
#endif
#include "Eigen/Geometry"
#ifdef _WIN32
#pragma warning(pop)
#endif

#include "vacancy/log.h"

namespace vacancy {

// borrow from glm
// radians
template <typename genType>
genType radians(genType degrees) {
  // "'radians' only accept floating-point input"
  assert(std::numeric_limits<genType>::is_iec559);

  return degrees * static_cast<genType>(0.01745329251994329576923690768489);
}

// degrees
template <typename genType>
genType degrees(genType radians) {
  // "'degrees' only accept floating-point input"
  assert(std::numeric_limits<genType>::is_iec559);

  return radians * static_cast<genType>(57.295779513082320876798154814105);
}

// https://stackoverflow.com/questions/13768423/setting-up-projection-model-and-view-transformations-for-vertex-shader-in-eige
template <typename T>
void c2w(const Eigen::Matrix<T, 3, 1>& position,
         const Eigen::Matrix<T, 3, 1>& target, const Eigen::Matrix<T, 3, 1>& up,
         Eigen::Matrix<T, 3, 3>* R) {
  assert(std::numeric_limits<T>::is_iec559);

  R->col(2) = (target - position).normalized();
  R->col(0) = R->col(2).cross(up).normalized();
  R->col(1) = R->col(2).cross(R->col(0));
}

template <typename genType>
void c2w(const Eigen::Matrix<genType, 3, 1>& position,
         const Eigen::Matrix<genType, 3, 1>& target,
         const Eigen::Matrix<genType, 3, 1>& up,
         Eigen::Matrix<genType, 4, 4>* T) {
  assert(std::numeric_limits<genType>::is_iec559);

  *T = Eigen::Matrix<genType, 4, 4>::Identity();

  Eigen::Matrix<genType, 3, 3> R;
  c2w(position, target, up, &R);

  T->topLeftCorner(3, 3) = R;
  T->topRightCorner(3, 1) = position;
}

template <typename T>
std::string zfill(const T& val, int num = 5) {
  std::ostringstream sout;
  sout << std::setfill('0') << std::setw(num) << val;
  return sout.str();
}

}  // namespace vacancy
