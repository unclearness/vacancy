/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>
#include <fstream>

#include "vacancy/voxel_carver.h"

std::vector<std::string> split(std::string& input, char delimiter) {
  std::istringstream stream(input);
  std::string field;
  std::vector<std::string> result;
  while (std::getline(stream, field, delimiter)) {
    result.push_back(field);
  }
  return result;
}

bool LoadTumFormat(const std::string& path,
                   std::vector<std::pair<int, Eigen::Affine3d>>* poses) {
  poses->clear();

  std::ifstream ifs(path);

  std::string line;
  while (std::getline(ifs, line)) {
    std::vector<std::string> splited = split(line, ' ');
    if (splited.size() != 8) {
      vacancy::LOGE("wrong tum format\n");
      return false;
    }

    std::pair<int, Eigen::Affine3d> pose;
    pose.first = std::atoi(splited[0].c_str());

    Eigen::Translation3d t;
    t.x() = std::atof(splited[1].c_str());
    t.y() = std::atof(splited[2].c_str());
    t.z() = std::atof(splited[3].c_str());

    Eigen::Quaterniond q;
    q.x() = std::atof(splited[4].c_str());
    q.y() = std::atof(splited[5].c_str());
    q.z() = std::atof(splited[6].c_str());
    q.w() = std::atof(splited[7].c_str());

    pose.second = t * q;

    poses->push_back(pose);
  }

  return true;
}

bool LoadTumFormat(const std::string& path,
                   std::vector<Eigen::Affine3d>* poses) {
  std::vector<std::pair<int, Eigen::Affine3d>> tmp_poses;
  bool ret = LoadTumFormat(path, &tmp_poses);
  if (!ret) {
    return false;
  }

  poses->clear();
  for (const auto& pose_pair : tmp_poses) {
    poses->push_back(pose_pair.second);
  }

  return true;
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir{"../data/"};
  std::vector<Eigen::Affine3d> poses;
  LoadTumFormat(data_dir + "tumpose.txt", &poses);

  vacancy::VoxelCarver carver;
  vacancy::VoxelCarverOption option;
  option.bb_min = Eigen::Vector3f(-250.000000, -344.586151, -129.982697);
  option.bb_max = Eigen::Vector3f(250.000000, 150.542343, 257.329224);

  float bb_offset = 20.0f;
  option.bb_min[0] -= bb_offset;
  option.bb_min[1] -= bb_offset;
  option.bb_min[2] -= bb_offset;

  option.bb_max[0] += bb_offset;
  option.bb_max[1] += bb_offset;
  option.bb_max[2] += bb_offset;

  option.resolution = 10.0f;
  option.debug_dir = data_dir;

  carver.set_option(option);

  carver.Init();

  float r = 0.5f;  // scale to smaller size from VGA
  int width = static_cast<int>(640 * r);
  int height = static_cast<int>(480 * r);
  Eigen::Vector2f principal_point(318.6f * r, 255.3f * r);
  Eigen::Vector2f focal_length(517.3f * r, 516.5f * r);
  std::shared_ptr<vacancy::Camera> camera =
      std::make_shared<vacancy::PinholeCamera>(width, height,
                                               Eigen::Affine3d::Identity(),
                                               principal_point, focal_length);

  for (size_t i = 0; i < 6; i++) {
    camera->set_c2w(poses[i]);

    std::string num = vacancy::zfill(i);

    vacancy::Image1b silhouette;
    silhouette.Load(data_dir + "/mask_" + num + ".png");

    vacancy::Image1f sdf;
    vacancy::Image3b vis_sdf;
    carver.Carve(*camera, silhouette, &sdf);

    vacancy::SignedDistance2Color(sdf, &vis_sdf, -100, 100);
    vis_sdf.WritePng(data_dir + "/sdf_" + num + ".png");

    vacancy::Mesh mesh;
    carver.ExtractVoxel(&mesh, true, true);
    mesh.WritePly(data_dir + "/voxel_" + num + ".ply");

    carver.ExtractIsoSurface(&mesh, 0.0);
    mesh.WritePly(data_dir + "/surface_" + num + ".ply");
  }

  return 0;
}
