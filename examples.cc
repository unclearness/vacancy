/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>
#include <fstream>

#include "vacancy/voxel_carver.h"

std::vector<std::string> Split(const std::string& input, char delimiter) {
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
    std::vector<std::string> splited = Split(line, ' ');
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

// test by bunny data with 6 views
int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  std::string data_dir{"../data/"};
  std::vector<Eigen::Affine3d> poses;
  LoadTumFormat(data_dir + "tumpose.txt", &poses);

  vacancy::VoxelCarver carver;
  vacancy::VoxelCarverOption option;

  // exact mesh bounding box computed in advacne
  option.bb_min = Eigen::Vector3f(-250.000000f, -344.586151f, -129.982697f);
  option.bb_max = Eigen::Vector3f(250.000000f, 150.542343f, 257.329224f);

  // add offset to the bounding box to keep boundary clean
  float bb_offset = 20.0f;
  option.bb_min[0] -= bb_offset;
  option.bb_min[1] -= bb_offset;
  option.bb_min[2] -= bb_offset;

  option.bb_max[0] += bb_offset;
  option.bb_max[1] += bb_offset;
  option.bb_max[2] += bb_offset;

  // voxel resolution is 10mm
  option.resolution = 10.0f;

  carver.set_option(option);

  carver.Init();

  // image size and intrinsic parameters
  int width = 320;
  int height = 240;
  Eigen::Vector2f principal_point(159.3f, 127.65f);
  Eigen::Vector2f focal_length(258.65f, 258.25f);
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
    // Carve() is the main process to update voxels. Corresponds to the fusion
    // step in KinectFusion
    carver.Carve(*camera, silhouette, &sdf);

    // save SDF visualization
    vacancy::Image3b vis_sdf;
    vacancy::SignedDistance2Color(sdf, &vis_sdf, -1.0f, 1.0f);
    vis_sdf.WritePng(data_dir + "/sdf_" + num + ".png");

    vacancy::Mesh mesh;
    // voxel extraction
    // slow for algorithm itself and saving to disk
    carver.ExtractVoxel(&mesh);
    mesh.WritePly(data_dir + "/voxel_" + num + ".ply");

    // marching cubes
    // smoother and faster
    carver.ExtractIsoSurface(&mesh, 0.0);
    mesh.WritePly(data_dir + "/surface_" + num + ".ply");
  }

  return 0;
}
