/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "currender/renderer.h"

#include <cassert>

#include "currender/pixel_shader.h"
#include "currender/timer.h"
#include "nanort/nanort.h"

namespace {
void PrepareRay(nanort::Ray<float>* ray, const Eigen::Vector3f& camera_pos_w,
                const Eigen::Vector3f& ray_w) {
  const float kFar = 1.0e+30f;
  ray->min_t = 0.0001f;
  ray->max_t = kFar;

  // camera position in world coordinate
  ray->org[0] = camera_pos_w[0];
  ray->org[1] = camera_pos_w[1];
  ray->org[2] = camera_pos_w[2];

  // ray in world coordinate
  ray->dir[0] = ray_w[0];
  ray->dir[1] = ray_w[1];
  ray->dir[2] = ray_w[2];
}
}  // namespace

namespace currender {

RendererOption::RendererOption() {}

RendererOption::~RendererOption() {}

void RendererOption::CopyTo(RendererOption* dst) const {
  dst->diffuse_color = diffuse_color;
  dst->depth_scale = depth_scale;
  dst->interp = interp;
  dst->shading_normal = shading_normal;
  dst->diffuse_shading = diffuse_shading;
  dst->backface_culling = backface_culling;
}

// Renderer::Impl implementation
class Renderer::Impl {
  bool mesh_initialized_{false};
  std::shared_ptr<const Camera> camera_{nullptr};
  std::shared_ptr<const Mesh> mesh_{nullptr};
  RendererOption option_;

  std::vector<float> flatten_vertices_;
  std::vector<unsigned int> flatten_faces_;

  nanort::BVHBuildOptions<float> build_options_;
  std::unique_ptr<nanort::TriangleMesh<float>> triangle_mesh_;
  std::unique_ptr<nanort::TriangleSAHPred<float>> triangle_pred_;
  nanort::BVHAccel<float> accel_;
  nanort::BVHBuildStatistics stats_;
  float bmin_[3], bmax_[3];

  bool ValidateAndInitBeforeRender(Image3b* color, Image1f* depth,
                                   Image3f* normal, Image1b* mask,
                                   Image1i* face_id) const;

 public:
  Impl();
  ~Impl();

  explicit Impl(const RendererOption& option);
  void set_option(const RendererOption& option);

  void set_mesh(std::shared_ptr<const Mesh> mesh);

  bool PrepareMesh();

  void set_camera(std::shared_ptr<const Camera> camera);

  bool Render(Image3b* color, Image1f* depth, Image3f* normal, Image1b* mask,
              Image1i* face_id) const;

  bool RenderColor(Image3b* color) const;
  bool RenderDepth(Image1f* depth) const;
  bool RenderNormal(Image3f* normal) const;
  bool RenderMask(Image1b* mask) const;
  bool RenderFaceId(Image1i* face_id) const;

  bool RenderW(Image3b* color, Image1w* depth, Image3f* normal, Image1b* mask,
               Image1i* face_id) const;
  bool RenderDepthW(Image1w* depth) const;
};

Renderer::Impl::Impl() {}
Renderer::Impl::~Impl() {}

Renderer::Impl::Impl(const RendererOption& option) { set_option(option); }

void Renderer::Impl::set_option(const RendererOption& option) {
  option.CopyTo(&option_);
}

void Renderer::Impl::set_mesh(std::shared_ptr<const Mesh> mesh) {
  mesh_initialized_ = false;
  mesh_ = mesh;

  if (mesh_->face_normals().empty()) {
    LOGW("face normal is empty. culling and shading may not work\n");
  }

  if (mesh_->normals().empty()) {
    LOGW("vertex normal is empty. shading may not work\n");
  }

  flatten_vertices_.clear();
  flatten_faces_.clear();

  const std::vector<Eigen::Vector3f>& vertices = mesh_->vertices();
  flatten_vertices_.resize(vertices.size() * 3);
  for (size_t i = 0; i < vertices.size(); i++) {
    flatten_vertices_[i * 3 + 0] = vertices[i][0];
    flatten_vertices_[i * 3 + 1] = vertices[i][1];
    flatten_vertices_[i * 3 + 2] = vertices[i][2];
  }

  const std::vector<Eigen::Vector3i>& vertex_indices = mesh_->vertex_indices();
  flatten_faces_.resize(vertex_indices.size() * 3);
  for (size_t i = 0; i < vertex_indices.size(); i++) {
    flatten_faces_[i * 3 + 0] = vertex_indices[i][0];
    flatten_faces_[i * 3 + 1] = vertex_indices[i][1];
    flatten_faces_[i * 3 + 2] = vertex_indices[i][2];
  }
}

bool Renderer::Impl::PrepareMesh() {
  if (mesh_ == nullptr) {
    LOGE("mesh has not been set\n");
    return false;
  }

  if (flatten_vertices_.empty() || flatten_faces_.empty()) {
    LOGE("mesh is empty\n");
    return false;
  }

  bool ret = false;
  build_options_.cache_bbox = false;

  LOGI("  BVH build option:\n");
  LOGI("    # of leaf primitives: %d\n", build_options_.min_leaf_primitives);
  LOGI("    SAH binsize         : %d\n", build_options_.bin_size);

  Timer<> timer;
  timer.Start();

  triangle_mesh_.reset(new nanort::TriangleMesh<float>(
      &flatten_vertices_[0], &flatten_faces_[0], sizeof(float) * 3));

  triangle_pred_.reset(new nanort::TriangleSAHPred<float>(
      &flatten_vertices_[0], &flatten_faces_[0], sizeof(float) * 3));

  LOGI("num_triangles = %llu\n",
       static_cast<uint64_t>(mesh_->vertex_indices().size()));
  // LOGI("faces = %p\n", mesh_->vertex_indices().size());

  ret = accel_.Build(static_cast<unsigned int>(mesh_->vertex_indices().size()),
                     *triangle_mesh_, *triangle_pred_, build_options_);

  if (!ret) {
    LOGE("BVH building failed\n");
    return false;
  }

  timer.End();
  LOGI("  BVH build time: %.1f msecs\n", timer.elapsed_msec());

  stats_ = accel_.GetStatistics();

  LOGI("  BVH statistics:\n");
  LOGI("    # of leaf   nodes: %d\n", stats_.num_leaf_nodes);
  LOGI("    # of branch nodes: %d\n", stats_.num_branch_nodes);
  LOGI("  Max tree depth     : %d\n", stats_.max_tree_depth);

  accel_.BoundingBox(bmin_, bmax_);
  LOGI("  Bmin               : %f, %f, %f\n", bmin_[0], bmin_[1], bmin_[2]);
  LOGI("  Bmax               : %f, %f, %f\n", bmax_[0], bmax_[1], bmax_[2]);

  mesh_initialized_ = true;

  return true;
}

void Renderer::Impl::set_camera(std::shared_ptr<const Camera> camera) {
  camera_ = camera;
}

bool Renderer::Impl::ValidateAndInitBeforeRender(Image3b* color, Image1f* depth,
                                                 Image3f* normal, Image1b* mask,
                                                 Image1i* face_id) const {
  if (camera_ == nullptr) {
    LOGE("camera has not been set\n");
    return false;
  }
  if (!mesh_initialized_) {
    LOGE("mesh has not been initialized\n");
    return false;
  }
  if (option_.backface_culling && mesh_->face_normals().empty()) {
    LOGE("specified back-face culling but face normal is empty.\n");
    return false;
  }
  if (option_.diffuse_color == DiffuseColor::kTexture &&
      mesh_->diffuse_tex().empty()) {
    LOGE("specified texture as diffuse color but texture is empty.\n");
    return false;
  }
  if (option_.diffuse_color == DiffuseColor::kVertex &&
      mesh_->vertex_colors().empty()) {
    LOGE(
        "specified vertex color as diffuse color but vertex color is empty.\n");
    return false;
  }
  if (option_.shading_normal == ShadingNormal::kFace &&
      mesh_->face_normals().empty()) {
    LOGE("specified face normal as shading normal but face normal is empty.\n");
    return false;
  }
  if (option_.shading_normal == ShadingNormal::kVertex &&
      mesh_->normals().empty()) {
    LOGE(
        "specified vertex normal as shading normal but vertex normal is "
        "empty.\n");
    return false;
  }
  if (color == nullptr && depth == nullptr && normal == nullptr &&
      mask == nullptr && face_id == nullptr) {
    LOGE("all arguments are nullptr. nothing to do\n");
    return false;
  }

  int width = camera_->width();
  int height = camera_->height();

  if (color != nullptr) {
    color->Init(width, height);
  }
  if (depth != nullptr) {
    depth->Init(width, height);
  }
  if (normal != nullptr) {
    normal->Init(width, height);
  }
  if (mask != nullptr) {
    mask->Init(width, height);
  }
  if (face_id != nullptr) {
    // initialize with -1 (no hit)
    face_id->Init(width, height, -1);
  }

  return true;
}

bool Renderer::Impl::Render(Image3b* color, Image1f* depth, Image3f* normal,
                            Image1b* mask, Image1i* face_id) const {
  if (!ValidateAndInitBeforeRender(color, depth, normal, mask, face_id)) {
    return false;
  }

  // make pixel shader
  std::unique_ptr<PixelShader> pixel_shader = PixelShaderFactory::Create(
      option_.diffuse_color, option_.interp, option_.diffuse_shading);

  OrenNayarParam oren_nayar_param(option_.oren_nayar_sigma);

  const Eigen::Matrix3f w2c_R = camera_->w2c().rotation().cast<float>();
  const Eigen::Vector3f w2c_t = camera_->w2c().translation().cast<float>();

  Timer<> timer;
  timer.Start();
#if defined(_OPENMP) && defined(CURRENDER_USE_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (int y = 0; y < camera_->height(); y++) {
    for (int x = 0; x < camera_->width(); x++) {
      // ray from camera position in world coordinate
      Eigen::Vector3f ray_w, org_ray_w;
      camera_->ray_w(static_cast<float>(x), static_cast<float>(y), &ray_w);
      camera_->org_ray_w(static_cast<float>(x), static_cast<float>(y),
                         &org_ray_w);
      nanort::Ray<float> ray;
      PrepareRay(&ray, org_ray_w, ray_w);

      // shoot ray
      nanort::TriangleIntersector<> triangle_intersector(
          &flatten_vertices_[0], &flatten_faces_[0], sizeof(float) * 3);
      nanort::TriangleIntersection<> isect;
      bool hit = accel_.Traverse(ray, triangle_intersector, &isect);

      if (!hit) {
        continue;
      }

#if 1
				      unsigned int fid = isect.prim_id;
      float u = isect.u;
      float v = isect.v;

      // back-face culling
      if (option_.backface_culling) {
        // back-face if face normal has same direction to ray
        if (mesh_->face_normals()[fid].dot(ray_w) > 0) {
          continue;
        }
      }

      // fill face id
      if (face_id != nullptr) {
        face_id->at(x, y, 0) = fid;
      }

      // fill mask
      if (mask != nullptr) {
        mask->at(x, y, 0) = 255;
      }

      // convert hit position to camera coordinate to get depth value
      if (depth != nullptr) {
        Eigen::Vector3f hit_pos_w = org_ray_w + ray_w * isect.t;
        Eigen::Vector3f hit_pos_c = w2c_R * hit_pos_w + w2c_t;
        assert(0.0f <= hit_pos_c[2]);  // depth should be positive
        depth->at(x, y, 0) = hit_pos_c[2] * option_.depth_scale;
      }

      // calculate shading normal
      Eigen::Vector3f shading_normal_w = Eigen::Vector3f::Zero();
      if (option_.shading_normal == ShadingNormal::kFace) {
        shading_normal_w = mesh_->face_normals()[fid];
      } else if (option_.shading_normal == ShadingNormal::kVertex) {
        // barycentric interpolation of normal
        const auto& normals = mesh_->normals();
        const auto& normal_indices = mesh_->normal_indices();
        shading_normal_w = (1.0f - u - v) * normals[normal_indices[fid][0]] +
                           u * normals[normal_indices[fid][1]] +
                           v * normals[normal_indices[fid][2]];
      }

      // set shading normal
      if (normal != nullptr) {
        Eigen::Vector3f shading_normal_c =
            w2c_R * shading_normal_w;  // rotate to camera coordinate
        for (int k = 0; k < 3; k++) {
          normal->at(x, y, k) = shading_normal_c[k];
        }
      }

      // delegate color calculation to pixel_shader
      if (color != nullptr) {
        Eigen::Vector3f light_dir = ray_w;  // emit light same as ray
        PixelShaderInput pixel_shader_input(color, x, y, u, v, fid, &ray_w,
                                            &light_dir, &shading_normal_w,
                                            &oren_nayar_param, mesh_);
        pixel_shader->Process(pixel_shader_input);


      }
#endif  // 0
    }
  }
  timer.End();
  LOGI("  Rendering main loop time: %.1f msecs\n", timer.elapsed_msec());

  return true;
}

bool Renderer::Impl::RenderColor(Image3b* color) const {
  return Render(color, nullptr, nullptr, nullptr, nullptr);
}

bool Renderer::Impl::RenderDepth(Image1f* depth) const {
  return Render(nullptr, depth, nullptr, nullptr, nullptr);
}

bool Renderer::Impl::RenderNormal(Image3f* normal) const {
  return Render(nullptr, nullptr, normal, nullptr, nullptr);
}

bool Renderer::Impl::RenderMask(Image1b* mask) const {
  return Render(nullptr, nullptr, nullptr, mask, nullptr);
}

bool Renderer::Impl::RenderFaceId(Image1i* face_id) const {
  return Render(nullptr, nullptr, nullptr, nullptr, face_id);
}

bool Renderer::Impl::RenderW(Image3b* color, Image1w* depth, Image3f* normal,
                             Image1b* mask, Image1i* face_id) const {
  if (depth == nullptr) {
    LOGE("depth is nullptr");
    return false;
  }

  Image1f f_depth;
  bool org_ret = Render(color, &f_depth, normal, mask, face_id);

  if (org_ret) {
    //f_depth.ConvertTo(depth);
  }

  return org_ret;
}

bool Renderer::Impl::RenderDepthW(Image1w* depth) const {
  return RenderW(nullptr, depth, nullptr, nullptr, nullptr);
}

// Renderer implementation
Renderer::Renderer() : pimpl(std::unique_ptr<Impl>(new Impl)) {}

Renderer::~Renderer() {}

Renderer::Renderer(const RendererOption& option)
    : pimpl(std::unique_ptr<Impl>(new Impl(option))) {}

void Renderer::set_option(const RendererOption& option) {
  pimpl->set_option(option);
}

void Renderer::set_mesh(std::shared_ptr<const Mesh> mesh) {
  pimpl->set_mesh(mesh);
}

bool Renderer::PrepareMesh() { return pimpl->PrepareMesh(); }

void Renderer::set_camera(std::shared_ptr<const Camera> camera) {
  pimpl->set_camera(camera);
}

bool Renderer::Render(Image3b* color, Image1f* depth, Image3f* normal,
                      Image1b* mask, Image1i* face_id) const {
  return pimpl->Render(color, depth, normal, mask, face_id);
}

bool Renderer::RenderColor(Image3b* color) const {
  return pimpl->RenderColor(color);
}

bool Renderer::RenderDepth(Image1f* depth) const {
  return pimpl->RenderDepth(depth);
}

bool Renderer::RenderNormal(Image3f* normal) const {
  return pimpl->RenderNormal(normal);
}

bool Renderer::RenderMask(Image1b* mask) const {
  return pimpl->RenderMask(mask);
}

bool Renderer::RenderFaceId(Image1i* face_id) const {
  return pimpl->RenderFaceId(face_id);
}

bool Renderer::RenderW(Image3b* color, Image1w* depth, Image3f* normal,
                       Image1b* mask, Image1i* face_id) const {
  return pimpl->RenderW(color, depth, normal, mask, face_id);
}

bool Renderer::RenderDepthW(Image1w* depth) const {
  return pimpl->RenderDepthW(depth);
}

}  // namespace currender
