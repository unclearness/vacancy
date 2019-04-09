/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "vacancy/image.h"

#include <algorithm>
#include <array>
#include <random>
#include <unordered_map>

#ifdef VACANCY_USE_STB
#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
#include "stb/stb_image.h"
#ifdef _WIN32
#pragma warning(pop)
#endif

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "stb/stb_image_write.h"
#ifdef _WIN32
#pragma warning(pop)
#endif
#endif

namespace vacancy {

void Depth2Gray(const Image1f& depth, Image1b* vis_depth, float min_d,
                float max_d) {
  assert(min_d < max_d);
  assert(vis_depth != nullptr);

  vis_depth->Init(depth.width(), depth.height());

  float inv_denom = 1.0f / (max_d - min_d);
  for (int y = 0; y < vis_depth->height(); y++) {
    for (int x = 0; x < vis_depth->width(); x++) {
      auto d = depth.at(x, y, 0);

      float norm_color = (d - min_d) * inv_denom;
      norm_color = std::min(std::max(norm_color, 0.0f), 1.0f);

      vis_depth->at(x, y, 0) = static_cast<uint8_t>(norm_color * 255);
    }
  }
}

void Normal2Color(const Image3f& normal, Image3b* vis_normal) {
  assert(vis_normal != nullptr);

  vis_normal->Init(normal.width(), normal.height());

  // Followed https://en.wikipedia.org/wiki/Normal_mapping
  // X: -1 to +1 :  Red: 0 to 255
  // Y: -1 to +1 :  Green: 0 to 255
  // Z: 0 to -1 :  Blue: 128 to 255
  for (int y = 0; y < vis_normal->height(); y++) {
    for (int x = 0; x < vis_normal->width(); x++) {
      vis_normal->at(x, y, 0) = static_cast<uint8_t>(
          std::round((normal.at(x, y, 0) + 1.0) * 0.5 * 255));
      vis_normal->at(x, y, 1) = static_cast<uint8_t>(
          std::round((normal.at(x, y, 1) + 1.0) * 0.5 * 255));
      vis_normal->at(x, y, 2) =
          static_cast<uint8_t>(std::round(-normal.at(x, y, 2) * 127.0) + 128);
    }
  }
}

void FaceId2RandomColor(const Image1i& face_id, Image3b* vis_face_id) {
  assert(vis_face_id != nullptr);

  vis_face_id->Init(face_id.width(), face_id.height(), 0);

  std::unordered_map<int, std::array<uint8_t, 3>> id2color;

  for (int y = 0; y < vis_face_id->height(); y++) {
    for (int x = 0; x < vis_face_id->width(); x++) {
      int fid = face_id.at(x, y, 0);
      if (fid < 0) {
        continue;
      }

      std::array<uint8_t, 3> color;
      auto iter = id2color.find(fid);
      if (iter != id2color.end()) {
        color = iter->second;
      } else {
        std::mt19937 mt(fid);
        // stl distribution depends on environment while mt19937 is independent.
        // so simply mod mt19937 value for random color reproducing the same
        // color in different environment.
        color[0] = static_cast<uint8_t>(mt() % 256);
        color[1] = static_cast<uint8_t>(mt() % 256);
        color[2] = static_cast<uint8_t>(mt() % 256);
        id2color[fid] = color;
      }

      vis_face_id->at(x, y, 0) = color[0];
      vis_face_id->at(x, y, 1) = color[1];
      vis_face_id->at(x, y, 2) = color[2];
    }
  }
}

}  // namespace vacancy
