/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "vacancy/camera.h"

namespace vacancy {

Camera::Camera()
    : width_(-1),
      height_(-1),
      c2w_(Eigen::Affine3d::Identity()),
      w2c_(Eigen::Affine3d::Identity()) {}

Camera::~Camera() {}

Camera::Camera(int width, int height)
    : width_(width),
      height_(height),
      c2w_(Eigen::Affine3d::Identity()),
      w2c_(Eigen::Affine3d::Identity()) {}

Camera::Camera(int width, int height, const Eigen::Affine3d& c2w)
    : width_(width), height_(height), c2w_(c2w), w2c_(c2w_.inverse()) {}

int Camera::width() const { return width_; }

int Camera::height() const { return height_; }

const Eigen::Affine3d& Camera::c2w() const { return c2w_; }

const Eigen::Affine3d& Camera::w2c() const { return w2c_; }

void Camera::set_size(int width, int height) {
  width_ = width;
  height_ = height;
}
void Camera::set_c2w(const Eigen::Affine3d& c2w) {
  c2w_ = c2w;
  w2c_ = c2w_.inverse();
}

PinholeCamera::PinholeCamera()
    : Camera(), principal_point_(-1, -1), focal_length_(-1, -1) {}

PinholeCamera::~PinholeCamera() {}

PinholeCamera::PinholeCamera(int width, int height)
    : Camera(width, height), principal_point_(-1, -1), focal_length_(-1, -1) {}

PinholeCamera::PinholeCamera(int width, int height, float fov_y_deg)
    : Camera(width, height) {
  principal_point_[0] = width_ * 0.5f - 0.5f;
  principal_point_[1] = height_ * 0.5f - 0.5f;

  set_fov_y(fov_y_deg);
}

PinholeCamera::PinholeCamera(int width, int height, const Eigen::Affine3d& c2w)
    : Camera(width, height, c2w),
      principal_point_(-1, -1),
      focal_length_(-1, -1) {}

PinholeCamera::PinholeCamera(int width, int height, const Eigen::Affine3d& c2w,
                             float fov_y_deg)
    : Camera(width, height, c2w) {
  principal_point_[0] = width_ * 0.5f - 0.5f;
  principal_point_[1] = height_ * 0.5f - 0.5f;

  set_fov_y(fov_y_deg);
}

PinholeCamera::PinholeCamera(int width, int height, const Eigen::Affine3d& c2w,
                             const Eigen::Vector2f& principal_point,
                             const Eigen::Vector2f& focal_length)
    : Camera(width, height, c2w),
      principal_point_(principal_point),
      focal_length_(focal_length) {}

float PinholeCamera::fov_x() const {
  return degrees<float>(2 * std::atan(width_ * 0.5f / focal_length_[0]));
}

float PinholeCamera::fov_y() const {
  return degrees<float>(2 * std::atan(height_ * 0.5f / focal_length_[1]));
}

const Eigen::Vector2f& PinholeCamera::principal_point() const {
  return principal_point_;
}

const Eigen::Vector2f& PinholeCamera::focal_length() const {
  return focal_length_;
}

void PinholeCamera::set_principal_point(
    const Eigen::Vector2f& principal_point) {
  principal_point_ = principal_point;
}

void PinholeCamera::set_focal_length(const Eigen::Vector2f& focal_length) {
  focal_length_ = focal_length;
}

void PinholeCamera::set_fov_x(float fov_x_deg) {
  // same focal length per pixel for x and y
  focal_length_[0] =
      width_ * 0.5f /
      static_cast<float>(std::tan(radians<float>(fov_x_deg) * 0.5));
  focal_length_[1] = focal_length_[0];
}

void PinholeCamera::set_fov_y(float fov_y_deg) {
  // same focal length per pixel for x and y
  focal_length_[1] =
      height_ * 0.5f /
      static_cast<float>(std::tan(radians<float>(fov_y_deg) * 0.5));
  focal_length_[0] = focal_length_[1];
}

void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
                            Eigen::Vector3f* image_p) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
  (*image_p)[2] = camera_p[2];
}

void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
                            Eigen::Vector2f* image_p) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
}

void PinholeCamera::Project(const Eigen::Vector3f& camera_p,
                            Eigen::Vector2f* image_p, float* d) const {
  (*image_p)[0] =
      focal_length_[0] / camera_p[2] * camera_p[0] + principal_point_[0];
  (*image_p)[1] =
      focal_length_[1] / camera_p[2] * camera_p[1] + principal_point_[1];
  *d = camera_p[2];
}

void PinholeCamera::Unproject(const Eigen::Vector3f& image_p,
                              Eigen::Vector3f* camera_p) const {
  (*camera_p)[0] =
      (image_p[0] - principal_point_[0]) * image_p[2] / focal_length_[0];
  (*camera_p)[1] =
      (image_p[1] - principal_point_[1]) * image_p[2] / focal_length_[1];
  (*camera_p)[2] = image_p[2];
}

void PinholeCamera::Unproject(const Eigen::Vector2f& image_p, float d,
                              Eigen::Vector3f* camera_p) const {
  (*camera_p)[0] = (image_p[0] - principal_point_[0]) * d / focal_length_[0];
  (*camera_p)[1] = (image_p[1] - principal_point_[1]) * d / focal_length_[1];
  (*camera_p)[2] = d;
}

void PinholeCamera::org_ray_c(float x, float y, Eigen::Vector3f* org) const {
  (void)x;
  (void)y;
  (*org)[0] = 0.0f;
  (*org)[1] = 0.0f;
  (*org)[2] = 0.0f;
}

void PinholeCamera::org_ray_w(float x, float y, Eigen::Vector3f* org) const {
  (void)x;
  (void)y;
  *org = c2w_.matrix().block<3, 1>(0, 3).cast<float>();
}

void PinholeCamera::ray_c(float x, float y, Eigen::Vector3f* dir) const {
  (*dir)[0] = (x - principal_point_[0]) / focal_length_[0];
  (*dir)[1] = (y - principal_point_[1]) / focal_length_[1];
  (*dir)[2] = 1.0f;
  dir->normalize();
}

void PinholeCamera::ray_w(float x, float y, Eigen::Vector3f* dir) const {
  ray_c(x, y, dir);
  *dir = c2w_.matrix().block<3, 3>(0, 0).cast<float>() * *dir;
}

OrthoCamera::OrthoCamera() : Camera() {}
OrthoCamera::~OrthoCamera() {}
OrthoCamera::OrthoCamera(int width, int height) : Camera(width, height) {}
OrthoCamera::OrthoCamera(int width, int height, const Eigen::Affine3d& c2w)
    : Camera(width, height, c2w) {}

void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
                          Eigen::Vector3f* image_p) const {
  *image_p = camera_p;
}

void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
                          Eigen::Vector2f* image_p) const {
  (*image_p)[0] = camera_p[0];
  (*image_p)[1] = camera_p[1];
}

void OrthoCamera::Project(const Eigen::Vector3f& camera_p,
                          Eigen::Vector2f* image_p, float* d) const {
  (*image_p)[0] = camera_p[0];
  (*image_p)[1] = camera_p[1];
  *d = camera_p[2];
}

void OrthoCamera::Unproject(const Eigen::Vector3f& image_p,
                            Eigen::Vector3f* camera_p) const {
  *camera_p = image_p;
}

void OrthoCamera::Unproject(const Eigen::Vector2f& image_p, float d,
                            Eigen::Vector3f* camera_p) const {
  (*camera_p)[0] = image_p[0];
  (*camera_p)[1] = image_p[1];
  (*camera_p)[2] = d;
}

void OrthoCamera::org_ray_c(float x, float y, Eigen::Vector3f* org) const {
  (*org)[0] = x - width_ / 2;
  (*org)[1] = y - height_ / 2;
  (*org)[2] = 0.0f;
}

void OrthoCamera::org_ray_w(float x, float y, Eigen::Vector3f* org) const {
  *org = c2w_.matrix().block<3, 1>(0, 3).cast<float>();

  Eigen::Vector3f x_direc =
      c2w_.matrix().block<3, 3>(0, 0).col(0).cast<float>();
  Eigen::Vector3f y_direc =
      c2w_.matrix().block<3, 3>(0, 0).col(1).cast<float>();

  Eigen::Vector3f offset_x = (x - width_ * 0.5f) * x_direc;
  Eigen::Vector3f offset_y = (y - height_ * 0.5f) * y_direc;

  *org += offset_x;
  *org += offset_y;
}

void OrthoCamera::ray_c(float x, float y, Eigen::Vector3f* dir) const {
  (void)x;
  (void)y;
  // parallell ray along with z axis
  (*dir)[0] = 0.0f;
  (*dir)[1] = 0.0f;
  (*dir)[2] = 1.0f;
}

void OrthoCamera::ray_w(float x, float y, Eigen::Vector3f* dir) const {
  (void)x;
  (void)y;
  // extract z direction of camera pose
  *dir = Eigen::Vector3f(c2w_.matrix().block<3, 3>(0, 0).cast<float>().col(2));
}

}  // namespace vacancy
