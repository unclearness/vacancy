/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vacancy/common.h"

#ifdef VACANCY_USE_STB
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#endif

namespace vacancy {

template <typename T, int N>
class Image {
  std::vector<T> data_;
  int width_{-1};
  int height_{-1};
  const int bit_depth_{sizeof(T)};
  const int channel_{N};

 public:
  Image() {}
  ~Image() {}
  Image(int width, int height) { Init(width, height); }
  Image(int width, int height, T val) { Init(width, height, val); }
  int width() const { return width_; }
  int height() const { return height_; }
  int channel() const { return channel_; }
  const std::vector<T>& data() const { return data_; }
  std::vector<T>* data_ptr() { return &data_; }
  void Clear() {
    data_.clear();
    width_ = -1;
    height_ = -1;
  }
  bool empty() const {
    if (width_ < 0 || height_ < 0 || data_.empty()) {
      return true;
    }
    return false;
  }
  void Init(int width, int height, T val = 0) {
    Clear();
    width_ = width;
    height_ = height;
    data_.resize(height_ * width_ * channel_, val);
  }
  T* at(int x, int y) {
    assert(0 <= x && x < width_ && 0 <= y && y < height_);
    return &data_[0] + (width_ * channel_ * y + x * channel_);
  }
  const T* at(int x, int y) const {
    assert(0 <= x && x < width_ && 0 <= y && y < height_);
    return &data_[0] + (width_ * channel_ * y + x * channel_);
  }
  T& at(int x, int y, int c) {
    assert(0 <= x && x < width_ && 0 <= y && y < height_ && 0 <= c &&
           c < channel_);
    return data_[width_ * channel_ * y + x * channel_ + c];
  }
  const T& at(int x, int y, int c) const {
    assert(0 <= x && x < width_ && 0 <= y && y < height_ && 0 <= c &&
           c < channel_);
    return data_[width_ * channel_ * y + x * channel_ + c];
  }

#ifdef VACANCY_USE_STB
  bool Load(const std::string& path) {
    unsigned char* in_pixels_tmp;
    int width;
    int height;
    int bpp;

    in_pixels_tmp = stbi_load(path.c_str(), &width, &height, &bpp, 0);

    if (bpp != channel_) {
      delete in_pixels_tmp;
      LOGE("desired channel %d, actual %d\n", channel_, bpp);
      return false;
    }

    width_ = width;
    height_ = height;

    data_.resize(static_cast<size_t>(height_) * static_cast<size_t>(width_) *
                 static_cast<size_t>(channel_));
    std::memcpy(&data_[0], in_pixels_tmp,
                sizeof(T) * channel_ * width_ * height_);

    delete in_pixels_tmp;
    return true;
  }

  bool WritePng(const std::string& path) const {
    if (bit_depth_ != 1) {
      LOGE("1 byte per channel is required to save by stb_image: actual %d\n",
           bit_depth_);
      return false;
    }

    if (width_ < 0 || height_ < 0) {
      LOGE("image is empty\n");
      return false;
    }

    stbi_write_png(path.c_str(), width_, height_, channel_, &data_[0],
                   width_ * channel_ * sizeof(T));
    return true;
  }
#else
  bool Load(const std::string& path) {
    (void)path;
    LOGE("This method is not supported by the current configuration\n");
    return false;
  }
  bool WritePng(const std::string& path) const {
    (void)path;
    LOGE("This method is not supported by the current configuration\n");
    return false;
  }
#endif

  template <typename TT, int NN>
  bool ConvertTo(Image<TT, NN>* dst, float scale = 1.0f) const {
    if (channel_ != dst->channel()) {
      LOGE("ConvertTo failed src channel %d, dst channel %d\n", channel_,
           dst->channel());
      return false;
    }

    dst->Init(width_, height_);

    for (int y = 0; y < height_; y++) {
      for (int x = 0; x < width_; x++) {
        for (int c = 0; c < N; c++) {
          dst->at(x, y, c) = static_cast<TT>(scale * at(x, y, c));
        }
      }
    }
    return true;
  }

  bool CopyTo(Image<T, N>* dst) const { return ConvertTo(dst, 1.0f); }
};

using Image1b = Image<uint8_t, 1>;   // For gray image.
using Image3b = Image<uint8_t, 3>;   // For color image. RGB order.
using Image1w = Image<uint16_t, 1>;  // For depth image with 16 bit (unsigned
                                     // short) mm-scale format
using Image1i =
    Image<int32_t, 1>;  // For face visibility. face id is within int32_t
using Image1f = Image<float, 1>;  // For depth image with any scale
using Image3f = Image<float, 3>;  // For normal or point cloud. XYZ order.

void Depth2Gray(const Image1f& depth, Image1b* vis_depth, float min_d = 200.0f,
                float max_d = 1500.0f);

void Normal2Color(const Image3f& normal, Image3b* vis_normal);

void FaceId2RandomColor(const Image1i& face_id, Image3b* vis_face_id);

}  // namespace vacancy
