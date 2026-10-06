#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/QR>
#include <fcl/fcl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Matrix3 = Eigen::Matrix3d;
using Vector3 = Eigen::Vector3d;

struct Transform {
  Matrix3 rotation = Matrix3::Identity();
  Vector3 translation = Vector3::Zero();
};

Transform compose(const Transform& left, const Transform& right) {
  return {
      left.rotation * right.rotation,
      left.translation + left.rotation * right.translation,
  };
}

Transform inverse(const Transform& value) {
  const Matrix3 rotation = value.rotation.transpose();
  return {rotation, -(rotation * value.translation)};
}

Matrix3 read_matrix(const double* values) {
  Matrix3 result;
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      result(row, column) = values[row * 3 + column];
    }
  }
  return result;
}

Vector3 read_vector(const double* values) {
  return {values[0], values[1], values[2]};
}

void write_matrix(double* values, const Matrix3& matrix) {
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      values[row * 3 + column] = matrix(row, column);
    }
  }
}

void write_vector(double* values, const Vector3& vector) {
  values[0] = vector.x();
  values[1] = vector.y();
  values[2] = vector.z();
}

Vector3 rotation_log(const Matrix3& rotation) {
  const Vector3 vector{
      rotation(2, 1) - rotation(1, 2),
      rotation(0, 2) - rotation(2, 0),
      rotation(1, 0) - rotation(0, 1),
  };
  const double sine = 0.5 * vector.norm();
  const double cosine = std::clamp(
      0.5 * (rotation.trace() - 1.0), -1.0, 1.0);
  if (sine <= 1.0e-12) {
    return 0.5 * vector;
  }
  const double angle = std::atan2(sine, cosine);
  return (angle / (2.0 * sine)) * vector;
}

Matrix3 rotation_from_vector(const Vector3& value) {
  const double angle = value.norm();
  if (angle <= 1.0e-15) {
    return Matrix3::Identity();
  }
  return Eigen::AngleAxisd(angle, value / angle).toRotationMatrix();
}

Vector3 bounded_vector(const Vector3& value, double maximum_norm) {
  const double norm = value.norm();
  if (norm <= maximum_norm || norm <= 1.0e-15) {
    return value;
  }
  return value * (maximum_norm / norm);
}

fcl::Transform3d to_fcl(const Transform& value) {
  fcl::Transform3d result = fcl::Transform3d::Identity();
  result.linear() = value.rotation;
  result.translation() = value.translation;
  return result;
}

std::uint64_t instance_key(int module, int geometry) {
  return (static_cast<std::uint64_t>(
              static_cast<std::uint32_t>(module))
          << 32) |
         static_cast<std::uint32_t>(geometry);
}

template <typename T>
T read_binary_value(std::ifstream* stream, const char* label) {
  T value{};
  stream->read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!*stream) {
    throw std::runtime_error(
        std::string("truncated binary STL while reading ") + label);
  }
  return value;
}

std::shared_ptr<fcl::BVHModel<fcl::OBBRSSd>> load_binary_stl(
    const std::string& path,
    const Vector3& scale) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open collision mesh: " + path);
  }
  char header[80];
  stream.read(header, sizeof(header));
  if (!stream) {
    throw std::runtime_error("collision STL header is truncated: " + path);
  }
  const std::uint32_t triangle_count =
      read_binary_value<std::uint32_t>(&stream, "triangle count");
  stream.seekg(0, std::ios::end);
  const std::streamoff byte_count = stream.tellg();
  const std::streamoff expected =
      84 + static_cast<std::streamoff>(triangle_count) * 50;
  if (byte_count != expected) {
    throw std::runtime_error(
        "collision mesh must be binary STL for posture collision: " + path);
  }
  stream.seekg(84, std::ios::beg);
  std::vector<fcl::Vector3d> vertices;
  std::vector<fcl::Triangle> triangles;
  vertices.reserve(static_cast<std::size_t>(triangle_count) * 3);
  triangles.reserve(triangle_count);
  for (std::uint32_t triangle = 0; triangle < triangle_count; ++triangle) {
    for (int component = 0; component < 3; ++component) {
      (void)read_binary_value<float>(&stream, "normal");
    }
    const int first = static_cast<int>(vertices.size());
    for (int vertex = 0; vertex < 3; ++vertex) {
      const double x =
          static_cast<double>(read_binary_value<float>(&stream, "vertex x"));
      const double y =
          static_cast<double>(read_binary_value<float>(&stream, "vertex y"));
      const double z =
          static_cast<double>(read_binary_value<float>(&stream, "vertex z"));
      vertices.emplace_back(
          x * scale.x(), y * scale.y(), z * scale.z());
    }
    triangles.emplace_back(first, first + 1, first + 2);
    (void)read_binary_value<std::uint16_t>(&stream, "attribute bytes");
  }
  auto model = std::make_shared<fcl::BVHModel<fcl::OBBRSSd>>();
  if (model->beginModel(
          static_cast<int>(triangles.size()),
          static_cast<int>(vertices.size())) != fcl::BVH_OK ||
      model->addSubModel(vertices, triangles) != fcl::BVH_OK ||
      model->endModel() != fcl::BVH_OK) {
    throw std::runtime_error("FCL could not build collision BVH: " + path);
  }
  return model;
}

std::shared_ptr<fcl::BVHModel<fcl::OBBRSSd>> cached_binary_stl(
    const std::string& path,
    const Vector3& scale) {
  static std::unordered_map<
      std::string,
      std::shared_ptr<fcl::BVHModel<fcl::OBBRSSd>>>
      cache;
  const std::string key =
      path + "|" + std::to_string(scale.x()) + "|" +
      std::to_string(scale.y()) + "|" + std::to_string(scale.z());
  const auto found = cache.find(key);
  if (found != cache.end()) {
    return found->second;
  }
  auto loaded = load_binary_stl(path, scale);
  cache[key] = loaded;
  return loaded;
}

double fcl_distance(
    const std::shared_ptr<fcl::CollisionGeometryd>& first,
    const Transform& first_pose,
    const std::shared_ptr<fcl::CollisionGeometryd>& second,
    const Transform& second_pose) {
  fcl::CollisionObjectd first_object(first, to_fcl(first_pose));
  fcl::CollisionObjectd second_object(second, to_fcl(second_pose));
  fcl::DistanceRequestd request;
  request.enable_signed_distance = true;
  request.enable_nearest_points = false;
  fcl::DistanceResultd result;
  const double distance =
      fcl::distance(&first_object, &second_object, request, result);
  if (!std::isfinite(distance)) {
    throw std::runtime_error("FCL returned a non-finite distance");
  }
  return distance;
}

bool fcl_collides(
    const std::shared_ptr<fcl::CollisionGeometryd>& first,
    const Transform& first_pose,
    const std::shared_ptr<fcl::CollisionGeometryd>& second,
    const Transform& second_pose) {
  fcl::CollisionObjectd first_object(first, to_fcl(first_pose));
  fcl::CollisionObjectd second_object(second, to_fcl(second_pose));
  fcl::CollisionRequestd request(1, false);
  fcl::CollisionResultd result;
  fcl::collide(&first_object, &second_object, request, result);
  return result.isCollision();
}

using WorldAabb = std::pair<Vector3, Vector3>;

WorldAabb box_world_aabb(
    const Transform& pose, const Vector3& half_extent) {
  const Vector3 world_half =
      pose.rotation.cwiseAbs() * half_extent;
  return {
      pose.translation - world_half,
      pose.translation + world_half};
}

double aabb_clearance(
    const WorldAabb& first,
    const WorldAabb& second) {
  Vector3 separation = Vector3::Zero();
  for (int axis = 0; axis < 3; ++axis) {
    separation[axis] = std::max(
        {0.0,
         second.first[axis] - first.second[axis],
         first.first[axis] - second.second[axis]});
  }
  return separation.norm();
}

struct Joint {
  int parent_link = -1;
  int child_link = -1;
  int type = 0;
  int q_index = -1;
  Transform origin;
  Vector3 axis = Vector3::UnitX();
};

struct Edge {
  int parent_module = -1;
  int child_module = -1;
  int parent_link = -1;
  int child_link = -1;
  Transform relation;
};

struct AnchorSpec {
  int module = -1;
  int link = -1;
  Transform local;
  Transform target;
};

struct Evaluation {
  std::vector<Transform> roots;
  std::vector<Transform> links;
  std::vector<Transform> anchors;
  Vector3 com = Vector3::Zero();
};

struct SolverConfig {
  int maximum_iterations = 60;
  int relaxed_seed_maximum_iterations = 40;
  int feasible_refinement_iterations = 3;
  double damping = 2.0e-3;
  double position_weight = 1.0;
  double attitude_weight = 0.35;
  double continuity_regularization_weight = 1.0e-5;
  double pitch_joint_regularization_weight = 1.0e-2;
  double finite_difference_step_rad = 1.0e-5;
  double maximum_joint_step_rad = 0.20;
  double maximum_base_translation_step_m = 0.20;
  double maximum_base_rotation_step_rad = 0.20;
  double anchor_position_tolerance_m = 0.005;
  double anchor_attitude_tolerance_rad = 0.10;
  double collision_margin_m = 0.005;
  double collision_feasibility_tolerance_m = 0.0002;
  bool use_relaxed_seed = true;
  double collision_activation_distance_m = 0.025;
  double collision_weight = 4.0;
  int collision_refinement_iterations = 20;
  int collision_line_search_steps = 8;
};

struct NativeSolution {
  bool feasible = false;
  Eigen::VectorXd q;
  Transform base;
  Evaluation evaluation;
  double maximum_position_error = 0.0;
  double maximum_attitude_error = 0.0;
  double minimum_proxy_clearance = std::numeric_limits<double>::infinity();
  int active_proxy_pair_count = 0;
  int iterations = 0;
};

struct CollisionGeometry {
  int link = -1;
  Transform exact_local;
  Transform proxy_local;
  Vector3 half_extent = Vector3::Zero();
  std::string mesh_path;
  Vector3 mesh_scale = Vector3::Ones();
  std::shared_ptr<fcl::Convexd> proxy;
  std::shared_ptr<fcl::BVHModel<fcl::OBBRSSd>> exact;
};

enum class CollisionTargetKind {
  kRobot,
  kObject,
};

struct CollisionPair {
  int first_instance = -1;
  CollisionTargetKind second_kind = CollisionTargetKind::kRobot;
  int second_instance = -1;
  bool proxy_enabled = true;
};

struct CollisionScene {
  bool enabled = false;
  Transform object_pose;
  Vector3 object_size = Vector3::Zero();
  std::shared_ptr<fcl::Boxd> object_shape;
  std::unordered_set<std::uint64_t> allowed_object_instances;
  std::vector<CollisionPair> pairs;
  std::vector<CollisionPair> selected_contact_pairs;
};

struct CollisionFrameCache {
  std::vector<Transform> proxy_poses;
  std::vector<WorldAabb> proxy_aabbs;
  std::vector<Transform> exact_poses;
  WorldAabb object_aabb;
};

struct CollisionMetrics {
  std::vector<double> clearances;
  std::vector<bool> colliding;
  double minimum = std::numeric_limits<double>::infinity();
  int active_count = 0;
  int colliding_count = 0;
  int evaluated_count = 0;
  int narrow_phase_count = 0;
  int broad_phase_pruned_count = 0;
};

struct CachedRobotPairMetric {
  double clearance;
  bool colliding;
  bool broad_phase_safe;
};

template <typename T, int Flags>
void require_rank(
    const py::array_t<T, Flags>& value, int rank, const char* name) {
  if (value.ndim() != rank) {
    throw std::invalid_argument(std::string(name) + " has the wrong rank");
  }
}

class Kernel {
 public:
  Kernel(
      int module_count,
      int link_count,
      int local_joint_count,
      int root_link,
      int base_link,
      int base_module,
      py::array_t<int, py::array::c_style | py::array::forcecast> joint_parent,
      py::array_t<int, py::array::c_style | py::array::forcecast> joint_child,
      py::array_t<int, py::array::c_style | py::array::forcecast> joint_type,
      py::array_t<int, py::array::c_style | py::array::forcecast> joint_q_index,
      py::array_t<double, py::array::c_style | py::array::forcecast> joint_origin_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> joint_origin_p,
      py::array_t<double, py::array::c_style | py::array::forcecast> joint_axis,
      py::array_t<int, py::array::c_style | py::array::forcecast> edge_parent_module,
      py::array_t<int, py::array::c_style | py::array::forcecast> edge_child_module,
      py::array_t<int, py::array::c_style | py::array::forcecast> edge_parent_link,
      py::array_t<int, py::array::c_style | py::array::forcecast> edge_child_link,
      py::array_t<double, py::array::c_style | py::array::forcecast> edge_relation_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> edge_relation_p,
      py::array_t<double, py::array::c_style | py::array::forcecast> link_mass,
      py::array_t<double, py::array::c_style | py::array::forcecast> link_com)
      : module_count_(module_count),
        link_count_(link_count),
        local_joint_count_(local_joint_count),
        root_link_(root_link),
        base_link_(base_link),
        base_module_(base_module) {
    if (module_count_ <= 0 || link_count_ <= 0 || local_joint_count_ <= 0) {
      throw std::invalid_argument("kernel dimensions must be positive");
    }
    require_rank(joint_parent, 1, "joint_parent");
    require_rank(joint_child, 1, "joint_child");
    require_rank(joint_type, 1, "joint_type");
    require_rank(joint_q_index, 1, "joint_q_index");
    require_rank(joint_origin_r, 3, "joint_origin_r");
    require_rank(joint_origin_p, 2, "joint_origin_p");
    require_rank(joint_axis, 2, "joint_axis");
    const py::ssize_t joint_count = joint_parent.shape(0);
    if (joint_child.shape(0) != joint_count ||
        joint_type.shape(0) != joint_count ||
        joint_q_index.shape(0) != joint_count ||
        joint_origin_r.shape(0) != joint_count ||
        joint_origin_p.shape(0) != joint_count ||
        joint_axis.shape(0) != joint_count) {
      throw std::invalid_argument("joint arrays have inconsistent lengths");
    }
    auto parents = joint_parent.unchecked<1>();
    auto children = joint_child.unchecked<1>();
    auto types = joint_type.unchecked<1>();
    auto q_indices = joint_q_index.unchecked<1>();
    auto origins_r = joint_origin_r.unchecked<3>();
    auto origins_p = joint_origin_p.unchecked<2>();
    auto axes = joint_axis.unchecked<2>();
    joints_.reserve(joint_count);
    for (py::ssize_t index = 0; index < joint_count; ++index) {
      Joint joint;
      joint.parent_link = parents(index);
      joint.child_link = children(index);
      joint.type = types(index);
      joint.q_index = q_indices(index);
      for (int row = 0; row < 3; ++row) {
        joint.origin.translation[row] = origins_p(index, row);
        joint.axis[row] = axes(index, row);
        for (int column = 0; column < 3; ++column) {
          joint.origin.rotation(row, column) =
              origins_r(index, row, column);
        }
      }
      const double axis_norm = joint.axis.norm();
      if (axis_norm > 0.0) {
        joint.axis /= axis_norm;
      }
      joints_.push_back(joint);
    }

    require_rank(edge_parent_module, 1, "edge_parent_module");
    require_rank(edge_child_module, 1, "edge_child_module");
    require_rank(edge_parent_link, 1, "edge_parent_link");
    require_rank(edge_child_link, 1, "edge_child_link");
    require_rank(edge_relation_r, 3, "edge_relation_r");
    require_rank(edge_relation_p, 2, "edge_relation_p");
    const py::ssize_t edge_count = edge_parent_module.shape(0);
    auto parent_modules = edge_parent_module.unchecked<1>();
    auto child_modules = edge_child_module.unchecked<1>();
    auto parent_links = edge_parent_link.unchecked<1>();
    auto child_links = edge_child_link.unchecked<1>();
    auto relations_r = edge_relation_r.unchecked<3>();
    auto relations_p = edge_relation_p.unchecked<2>();
    edges_.reserve(edge_count);
    for (py::ssize_t index = 0; index < edge_count; ++index) {
      Edge edge;
      edge.parent_module = parent_modules(index);
      edge.child_module = child_modules(index);
      edge.parent_link = parent_links(index);
      edge.child_link = child_links(index);
      for (int row = 0; row < 3; ++row) {
        edge.relation.translation[row] = relations_p(index, row);
        for (int column = 0; column < 3; ++column) {
          edge.relation.rotation(row, column) =
              relations_r(index, row, column);
        }
      }
      edges_.push_back(edge);
    }

    require_rank(link_mass, 1, "link_mass");
    require_rank(link_com, 2, "link_com");
    if (link_mass.shape(0) != link_count_ ||
        link_com.shape(0) != link_count_) {
      throw std::invalid_argument("link arrays have inconsistent lengths");
    }
    auto masses = link_mass.unchecked<1>();
    auto centers = link_com.unchecked<2>();
    link_masses_.resize(link_count_);
    link_centers_.resize(link_count_);
    module_mass_ = 0.0;
    for (int index = 0; index < link_count_; ++index) {
      link_masses_[index] = masses(index);
      link_centers_[index] = {
          centers(index, 0), centers(index, 1), centers(index, 2)};
      module_mass_ += link_masses_[index];
    }
    if (module_mass_ <= 0.0) {
      throw std::invalid_argument("module mass must be positive");
    }
  }

  void configure_collision_geometry(
      py::array_t<int, py::array::c_style | py::array::forcecast>
          collision_link,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          collision_local_r,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          collision_local_p,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          proxy_center,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          proxy_half_extent,
      const py::list& convex_vertices,
      const py::list& convex_faces,
      const py::list& mesh_paths,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          mesh_scale,
      bool load_exact_geometry) {
    require_rank(collision_link, 1, "collision_link");
    require_rank(collision_local_r, 3, "collision_local_r");
    require_rank(collision_local_p, 2, "collision_local_p");
    require_rank(proxy_center, 2, "proxy_center");
    require_rank(proxy_half_extent, 2, "proxy_half_extent");
    require_rank(mesh_scale, 2, "mesh_scale");
    const py::ssize_t count = collision_link.shape(0);
    if (collision_local_r.shape(0) != count ||
        collision_local_p.shape(0) != count ||
        proxy_center.shape(0) != count ||
        proxy_half_extent.shape(0) != count ||
        mesh_scale.shape(0) != count ||
        static_cast<py::ssize_t>(py::len(convex_vertices)) != count ||
        static_cast<py::ssize_t>(py::len(convex_faces)) != count ||
        static_cast<py::ssize_t>(py::len(mesh_paths)) != count) {
      throw std::invalid_argument(
          "collision geometry arrays have inconsistent lengths");
    }
    auto links = collision_link.unchecked<1>();
    auto local_r = collision_local_r.unchecked<3>();
    auto local_p = collision_local_p.unchecked<2>();
    auto centers = proxy_center.unchecked<2>();
    auto half = proxy_half_extent.unchecked<2>();
    auto scales = mesh_scale.unchecked<2>();
    collision_configuration_valid_ = false;
    collision_geometry_.clear();
    nominal_same_module_collision_.clear();
    collision_pair_cache_.clear();
    exact_collision_geometry_loaded_ = load_exact_geometry;
    nominal_collision_pair_table_complete_ = false;
    collision_geometry_.reserve(count);
    for (py::ssize_t index = 0; index < count; ++index) {
      CollisionGeometry geometry;
      geometry.link = links(index);
      if (geometry.link < 0 || geometry.link >= link_count_) {
        throw std::invalid_argument(
            "collision geometry references an invalid link");
      }
      for (int row = 0; row < 3; ++row) {
        geometry.exact_local.translation[row] = local_p(index, row);
        geometry.half_extent[row] = half(index, row);
        geometry.mesh_scale[row] = scales(index, row);
        for (int column = 0; column < 3; ++column) {
          geometry.exact_local.rotation(row, column) =
              local_r(index, row, column);
        }
      }
      geometry.proxy_local = compose(
          geometry.exact_local,
          Transform{
              Matrix3::Identity(),
              Vector3{
                  centers(index, 0),
                  centers(index, 1),
                  centers(index, 2)}});
      if ((geometry.half_extent.array() <= 0.0).any()) {
        throw std::invalid_argument(
            "proxy half extents must be positive");
      }
      py::array_t<
          double,
          py::array::c_style | py::array::forcecast>
          vertices_array =
              py::cast<py::array>(convex_vertices[index]);
      py::array_t<
          int,
          py::array::c_style | py::array::forcecast>
          faces_array = py::cast<py::array>(convex_faces[index]);
      require_rank(vertices_array, 2, "convex_vertices item");
      require_rank(faces_array, 2, "convex_faces item");
      if (vertices_array.shape(1) != 3 ||
          faces_array.shape(1) != 3 ||
          vertices_array.shape(0) < 4 ||
          faces_array.shape(0) < 4) {
        throw std::invalid_argument(
            "convex hull arrays have invalid shape");
      }
      auto vertices_input = vertices_array.unchecked<2>();
      auto faces_input = faces_array.unchecked<2>();
      auto vertices =
          std::make_shared<std::vector<fcl::Vector3d>>();
      auto faces = std::make_shared<std::vector<int>>();
      vertices->reserve(vertices_array.shape(0));
      faces->reserve(faces_array.shape(0) * 4);
      for (py::ssize_t vertex = 0;
           vertex < vertices_array.shape(0);
           ++vertex) {
        vertices->emplace_back(
            vertices_input(vertex, 0),
            vertices_input(vertex, 1),
            vertices_input(vertex, 2));
      }
      for (py::ssize_t face = 0;
           face < faces_array.shape(0);
           ++face) {
        faces->push_back(3);
        faces->push_back(faces_input(face, 0));
        faces->push_back(faces_input(face, 1));
        faces->push_back(faces_input(face, 2));
      }
      geometry.mesh_path = py::cast<std::string>(mesh_paths[index]);
      geometry.proxy = std::make_shared<fcl::Convexd>(
          vertices,
          static_cast<int>(faces_array.shape(0)),
          faces,
          true);
      if (load_exact_geometry) {
        geometry.exact =
            cached_binary_stl(geometry.mesh_path, geometry.mesh_scale);
      }
      collision_geometry_.push_back(std::move(geometry));
    }
  }

  void configure_nominal_collision_pairs(
      py::array_t<int, py::array::c_style | py::array::forcecast>
          collision_pairs) {
    collision_configuration_valid_ = false;
    require_rank(collision_pairs, 2, "collision_pairs");
    if (collision_pairs.shape(1) != 2) {
      throw std::invalid_argument(
          "nominal collision pairs must have shape [N, 2]");
    }
    auto pairs = collision_pairs.unchecked<2>();
    nominal_same_module_collision_.clear();
    collision_pair_cache_.clear();
    for (py::ssize_t index = 0;
         index < collision_pairs.shape(0);
         ++index) {
      const int first = pairs(index, 0);
      const int second = pairs(index, 1);
      if (first < 0 || second < 0 ||
          first >= static_cast<int>(collision_geometry_.size()) ||
          second >= static_cast<int>(collision_geometry_.size()) ||
          first == second) {
        throw std::invalid_argument(
            "nominal collision pair index is invalid");
      }
      nominal_same_module_collision_[instance_key(
          std::min(first, second), std::max(first, second))] = true;
    }
    nominal_collision_pair_table_complete_ = true;
  }

  py::array_t<int> nominal_collision_pairs() const {
    if (!nominal_collision_pair_table_complete_) {
      throw std::runtime_error(
          "nominal collision pair table is incomplete");
    }
    std::vector<std::pair<int, int>> pairs;
    for (const auto& item : nominal_same_module_collision_) {
      if (!item.second) {
        continue;
      }
      pairs.emplace_back(
          static_cast<int>(item.first >> 32),
          static_cast<int>(item.first & 0xffffffffU));
    }
    std::sort(pairs.begin(), pairs.end());
    py::array_t<int> output(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(pairs.size()), 2});
    auto values = output.mutable_unchecked<2>();
    for (py::ssize_t index = 0;
         index < static_cast<py::ssize_t>(pairs.size());
         ++index) {
      values(index, 0) = pairs[index].first;
      values(index, 1) = pairs[index].second;
    }
    return output;
  }

  void set_collision_scene(
      bool object_enabled,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          object_r,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          object_p,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          object_size,
      py::array_t<int, py::array::c_style | py::array::forcecast>
          allowed_contact_module,
      py::array_t<int, py::array::c_style | py::array::forcecast>
          allowed_contact_link) {
    require_rank(object_r, 2, "object_r");
    require_rank(object_p, 1, "object_p");
    require_rank(object_size, 1, "object_size");
    require_rank(
        allowed_contact_module, 1, "allowed_contact_module");
    require_rank(allowed_contact_link, 1, "allowed_contact_link");
    if (object_r.shape(0) != 3 || object_r.shape(1) != 3 ||
        object_p.shape(0) != 3 || object_size.shape(0) != 3 ||
        allowed_contact_module.shape(0) !=
            allowed_contact_link.shape(0)) {
      throw std::invalid_argument("collision scene arrays have invalid shape");
    }
    if (collision_geometry_.empty()) {
      throw std::runtime_error(
          "collision geometry must be configured before a scene");
    }
    active_collision_scene_ = CollisionScene{};
    active_collision_scene_.enabled = true;
    active_collision_scene_.object_pose = {
        read_matrix(object_r.data()), read_vector(object_p.data())};
    active_collision_scene_.object_size = read_vector(object_size.data());
    if (object_enabled &&
        (active_collision_scene_.object_size.array() <= 0.0).any()) {
      throw std::invalid_argument(
          "enabled object collision size must be positive");
    }
    if (object_enabled) {
      active_collision_scene_.object_shape =
          std::make_shared<fcl::Boxd>(
              active_collision_scene_.object_size.x(),
              active_collision_scene_.object_size.y(),
              active_collision_scene_.object_size.z());
    }
    auto allowed_modules = allowed_contact_module.unchecked<1>();
    auto allowed_links = allowed_contact_link.unchecked<1>();
    for (py::ssize_t allowed = 0;
         allowed < allowed_contact_module.shape(0);
         ++allowed) {
      for (int geometry = 0;
           geometry < static_cast<int>(collision_geometry_.size());
           ++geometry) {
        if (collision_geometry_[geometry].link == allowed_links(allowed)) {
          active_collision_scene_.allowed_object_instances.insert(
              instance_key(allowed_modules(allowed), geometry));
        }
      }
    }
    std::vector<std::uint64_t> allowed_instances(
        active_collision_scene_.allowed_object_instances.begin(),
        active_collision_scene_.allowed_object_instances.end());
    std::sort(allowed_instances.begin(), allowed_instances.end());
    if (object_enabled) {
      active_collision_scene_.selected_contact_pairs.reserve(
          allowed_instances.size());
      for (std::uint64_t instance : allowed_instances) {
        const int module = static_cast<int>(instance >> 32);
        const int geometry = static_cast<int>(
            instance & std::numeric_limits<std::uint32_t>::max());
        active_collision_scene_.selected_contact_pairs.push_back(
            CollisionPair{
                module *
                        static_cast<int>(collision_geometry_.size()) +
                    geometry,
                CollisionTargetKind::kObject,
                -1});
      }
    }
    std::string pair_cache_key = object_enabled ? "1" : "0";
    for (std::uint64_t instance : allowed_instances) {
      pair_cache_key += "|" + std::to_string(instance);
    }
    const auto cached_pairs =
        collision_pair_cache_.find(pair_cache_key);
    if (cached_pairs != collision_pair_cache_.end()) {
      active_collision_scene_.pairs = cached_pairs->second;
    } else {
      build_collision_pairs(
          object_enabled, &active_collision_scene_);
      collision_pair_cache_.emplace(
          std::move(pair_cache_key),
          active_collision_scene_.pairs);
    }
  }

  void clear_collision_scene() {
    active_collision_scene_ = CollisionScene{};
  }

  // Same nonlinear posture residual as the readable NumPy reference, evaluated
  // with FK/CoM in one native batch (also used for finite differences).
  py::array_t<double> grasp_residual(
      py::array_t<double, py::array::c_style | py::array::forcecast> q,
      py::array_t<double, py::array::c_style | py::array::forcecast> offsets,
      py::array_t<double, py::array::c_style | py::array::forcecast> normalized,
      py::array_t<double, py::array::c_style | py::array::forcecast> body_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> body_p,
      py::array_t<int, py::array::c_style | py::array::forcecast> modules,
      py::array_t<int, py::array::c_style | py::array::forcecast> links,
      py::array_t<double, py::array::c_style | py::array::forcecast> local_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> local_p,
      py::array_t<double, py::array::c_style | py::array::forcecast> target_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> target_p,
      py::array_t<double, py::array::c_style | py::array::forcecast> nominal_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> nominal_p) const {
    require_rank(q,3,"q"); require_rank(offsets,2,"offsets"); require_rank(normalized,2,"normalized");
    require_rank(modules,1,"modules"); require_rank(links,1,"links");
    require_rank(body_r,2,"body rotation"); require_rank(body_p,1,"body position");
    const int count=q.shape(0), anchors_count=modules.shape(0), dimension=6+module_count_*local_joint_count_;
    if (q.shape(1)!=module_count_ || q.shape(2)!=local_joint_count_ || offsets.shape(0)!=count || normalized.shape(0)!=count || offsets.shape(1)!=dimension || normalized.shape(1)!=dimension || anchors_count<1 || links.shape(0)!=anchors_count || body_r.shape(0)!=3 || body_r.shape(1)!=3 || body_p.shape(0)!=3)
      throw std::invalid_argument("invalid grasp residual dimensions");
    for (const auto* a : {&local_r,&target_r,&nominal_r}) if (a->ndim()!=3 || a->shape(0)!=anchors_count || a->shape(1)!=3 || a->shape(2)!=3) throw std::invalid_argument("invalid grasp rotation array");
    for (const auto* a : {&local_p,&target_p,&nominal_p}) if (a->ndim()!=2 || a->shape(0)!=anchors_count || a->shape(1)!=3) throw std::invalid_argument("invalid grasp position array");
    std::vector<AnchorSpec> anchors(anchors_count);
    for(int a=0;a<anchors_count;++a) {
      if(modules.data()[a]<0 || modules.data()[a]>=module_count_ || links.data()[a]<0 || links.data()[a]>=link_count_) throw std::invalid_argument("invalid grasp anchor index");
      anchors[a].module=modules.data()[a];anchors[a].link=links.data()[a];
      anchors[a].local={read_matrix(local_r.data()+a*9),read_vector(local_p.data()+a*3)};
    }
    const Matrix3 rb=read_matrix(body_r.data()), nr0=read_matrix(nominal_r.data());
    const Vector3 bp=read_vector(body_p.data()), np0=read_vector(nominal_p.data());
    const int width=anchors_count*12+dimension;
    py::array_t<double> result({count,width});double* output=result.mutable_data();
    auto logarithm = [](const Matrix3& r) -> Vector3 {
      Eigen::Quaterniond quaternion(r); quaternion.normalize();
      if(quaternion.w()<0)quaternion.coeffs() *= -1.;
      const double length=quaternion.vec().norm();
      if(length<=1e-15)return 2.*quaternion.vec();
      return (2.*std::atan2(length,quaternion.w())/length)*quaternion.vec();
    };
    py::gil_scoped_release release;
    for(int s=0;s<count;++s) {
      const double* offset=offsets.data()+s*dimension;
      const Vector3 rv=read_vector(offset+3);const double angle=rv.norm();
      Matrix3 turn=Matrix3::Identity();if(angle>1e-15)turn=Eigen::AngleAxisd(angle,rv/angle).toRotationMatrix();
      const Transform body{turn*rb,bp+read_vector(offset)};
      Eigen::Map<const Eigen::VectorXd> joints(q.data()+s*(dimension-6),dimension-6);
      Evaluation evaluation=evaluate_one(joints,Transform{},anchors);
      for(auto& anchor : evaluation.anchors) {
        anchor.translation=body.translation+body.rotation*(anchor.translation-evaluation.com);
        anchor.rotation=body.rotation*anchor.rotation;
      }
      double* row=output+s*width;
      const Transform& first=evaluation.anchors[0];
      for(int a=0;a<anchors_count;++a) {
        const auto& actual=evaluation.anchors[a];
        write_vector(row+a*3,(actual.translation-read_vector(target_p.data()+a*3))/.001);
        write_vector(row+anchors_count*3+a*3,logarithm(actual.rotation*read_matrix(target_r.data()+a*9).transpose())/.01);
        const Vector3 relative_p=first.rotation.transpose()*(actual.translation-first.translation);
        const Vector3 nominal_relative_p=nr0.transpose()*(read_vector(nominal_p.data()+a*3)-np0);
        write_vector(row+anchors_count*6+a*3,(relative_p-nominal_relative_p)/.00005);
        const Matrix3 relative_r=first.rotation.transpose()*actual.rotation;
        const Matrix3 nominal_relative_r=nr0.transpose()*read_matrix(nominal_r.data()+a*9);
        write_vector(row+anchors_count*9+a*3,logarithm(relative_r*nominal_relative_r.transpose())/.0005);
      }
      for(int i=0;i<dimension;++i)row[anchors_count*12+i]=normalized.data()[s*dimension+i]*.1;
    }
    return result;
  }

  py::tuple evaluate(
      py::array_t<double, py::array::c_style | py::array::forcecast> q,
      py::array_t<double, py::array::c_style | py::array::forcecast> base_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> base_p,
      py::array_t<int, py::array::c_style | py::array::forcecast> anchor_module,
      py::array_t<int, py::array::c_style | py::array::forcecast> anchor_link,
      py::array_t<double, py::array::c_style | py::array::forcecast> anchor_local_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> anchor_local_p)
      const {
    require_rank(q, 3, "q");
    require_rank(base_r, 2, "base_r");
    require_rank(base_p, 1, "base_p");
    require_rank(anchor_module, 1, "anchor_module");
    require_rank(anchor_link, 1, "anchor_link");
    require_rank(anchor_local_r, 3, "anchor_local_r");
    require_rank(anchor_local_p, 2, "anchor_local_p");
    const py::ssize_t batch = q.shape(0);
    if (q.shape(1) != module_count_ ||
        q.shape(2) != local_joint_count_) {
      throw std::invalid_argument("q shape differs from kernel dimensions");
    }
    const py::ssize_t anchor_count = anchor_module.shape(0);
    auto q_values = q.unchecked<3>();
    auto anchor_modules = anchor_module.unchecked<1>();
    auto anchor_links = anchor_link.unchecked<1>();
    auto anchors_r = anchor_local_r.unchecked<3>();
    auto anchors_p = anchor_local_p.unchecked<2>();
    const Transform base{
        read_matrix(base_r.data()), read_vector(base_p.data())};

    py::array_t<double> output_root_r(
        std::vector<py::ssize_t>{batch, module_count_, 3, 3});
    py::array_t<double> output_root_p(
        std::vector<py::ssize_t>{batch, module_count_, 3});
    py::array_t<double> output_anchor_r(
        std::vector<py::ssize_t>{batch, anchor_count, 3, 3});
    py::array_t<double> output_anchor_p(
        std::vector<py::ssize_t>{batch, anchor_count, 3});
    py::array_t<double> output_com(
        std::vector<py::ssize_t>{batch, 3});
    auto root_r_mutable = output_root_r.mutable_unchecked<4>();
    auto root_p_mutable = output_root_p.mutable_unchecked<3>();
    auto anchor_r_mutable = output_anchor_r.mutable_unchecked<4>();
    auto anchor_p_mutable = output_anchor_p.mutable_unchecked<3>();
    auto com_mutable = output_com.mutable_unchecked<2>();

    const std::size_t local_count =
        static_cast<std::size_t>(module_count_ * link_count_);
    std::vector<Transform> local(local_count);
    std::vector<Transform> roots(module_count_);
    std::vector<Transform> link_root(link_count_);

    for (py::ssize_t sample = 0; sample < batch; ++sample) {
      for (int module = 0; module < module_count_; ++module) {
        std::fill(link_root.begin(), link_root.end(), Transform{});
        link_root[root_link_] = Transform{};
        for (const Joint& joint : joints_) {
          Transform motion;
          const double value =
              joint.q_index < 0 ? 0.0 : q_values(sample, module, joint.q_index);
          if (joint.type == 1) {
            motion.rotation =
                Eigen::AngleAxisd(value, joint.axis).toRotationMatrix();
          } else if (joint.type == 2) {
            motion.translation = value * joint.axis;
          }
          link_root[joint.child_link] = compose(
              compose(link_root[joint.parent_link], joint.origin), motion);
        }
        const Transform root_from_base = inverse(link_root[base_link_]);
        for (int link = 0; link < link_count_; ++link) {
          local[module * link_count_ + link] =
              compose(root_from_base, link_root[link]);
        }
      }

      roots.assign(module_count_, Transform{});
      roots[base_module_] = base;
      for (const Edge& edge : edges_) {
        const Transform parent_connect = compose(
            roots[edge.parent_module],
            local[edge.parent_module * link_count_ + edge.parent_link]);
        const Transform desired = compose(parent_connect, edge.relation);
        roots[edge.child_module] = compose(
            desired,
            inverse(local[
                edge.child_module * link_count_ + edge.child_link]));
      }

      Vector3 weighted_com = Vector3::Zero();
      for (int module = 0; module < module_count_; ++module) {
        write_matrix(
            &root_r_mutable(sample, module, 0, 0),
            roots[module].rotation);
        write_vector(
            &root_p_mutable(sample, module, 0),
            roots[module].translation);
        for (int link = 0; link < link_count_; ++link) {
          const Transform world = compose(
              roots[module], local[module * link_count_ + link]);
          weighted_com += link_masses_[link] *
                          (world.translation +
                           world.rotation * link_centers_[link]);
        }
      }
      const Vector3 assembled_com =
          weighted_com / (module_mass_ * module_count_);
      for (int axis = 0; axis < 3; ++axis) {
        com_mutable(sample, axis) = assembled_com[axis];
      }

      for (py::ssize_t anchor = 0; anchor < anchor_count; ++anchor) {
        Transform local_anchor;
        for (int row = 0; row < 3; ++row) {
          local_anchor.translation[row] = anchors_p(anchor, row);
          for (int column = 0; column < 3; ++column) {
            local_anchor.rotation(row, column) =
                anchors_r(anchor, row, column);
          }
        }
        const int module = anchor_modules(anchor);
        const int link = anchor_links(anchor);
        const Transform world = compose(
            compose(roots[module],
                    local[module * link_count_ + link]),
            local_anchor);
        write_matrix(
            &anchor_r_mutable(sample, anchor, 0, 0), world.rotation);
        write_vector(
            &anchor_p_mutable(sample, anchor, 0), world.translation);
      }
    }
    return py::make_tuple(
        output_root_r,
        output_root_p,
        output_anchor_r,
        output_anchor_p,
        output_com);
  }

  py::dict solve_centroidal(
      py::array_t<double, py::array::c_style | py::array::forcecast> initial_q,
      py::array_t<double, py::array::c_style | py::array::forcecast> lower,
      py::array_t<double, py::array::c_style | py::array::forcecast> upper,
      py::array_t<int, py::array::c_style | py::array::forcecast> pitch_mask,
      py::array_t<double, py::array::c_style | py::array::forcecast> centroidal_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> centroidal_p,
      py::array_t<int, py::array::c_style | py::array::forcecast> anchor_module,
      py::array_t<int, py::array::c_style | py::array::forcecast> anchor_link,
      py::array_t<double, py::array::c_style | py::array::forcecast> anchor_local_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> anchor_local_p,
      py::array_t<double, py::array::c_style | py::array::forcecast> target_r,
      py::array_t<double, py::array::c_style | py::array::forcecast> target_p,
      const py::dict& config_values) const {
    require_rank(initial_q, 2, "initial_q");
    require_rank(lower, 2, "lower");
    require_rank(upper, 2, "upper");
    require_rank(pitch_mask, 2, "pitch_mask");
    require_rank(centroidal_r, 2, "centroidal_r");
    require_rank(centroidal_p, 1, "centroidal_p");
    require_rank(anchor_module, 1, "anchor_module");
    require_rank(anchor_link, 1, "anchor_link");
    require_rank(anchor_local_r, 3, "anchor_local_r");
    require_rank(anchor_local_p, 2, "anchor_local_p");
    require_rank(target_r, 3, "target_r");
    require_rank(target_p, 2, "target_p");
    if (initial_q.shape(0) != module_count_ ||
        initial_q.shape(1) != local_joint_count_ ||
        lower.shape(0) != module_count_ ||
        lower.shape(1) != local_joint_count_ ||
        upper.shape(0) != module_count_ ||
        upper.shape(1) != local_joint_count_ ||
        pitch_mask.shape(0) != module_count_ ||
        pitch_mask.shape(1) != local_joint_count_) {
      throw std::invalid_argument("solver joint arrays have invalid shape");
    }
    const py::ssize_t anchor_count = anchor_module.shape(0);
    if (anchor_link.shape(0) != anchor_count ||
        anchor_local_r.shape(0) != anchor_count ||
        anchor_local_p.shape(0) != anchor_count ||
        target_r.shape(0) != anchor_count ||
        target_p.shape(0) != anchor_count) {
      throw std::invalid_argument("solver anchor arrays have invalid shape");
    }

    const int joint_count = module_count_ * local_joint_count_;
    Eigen::VectorXd initial(joint_count);
    Eigen::VectorXd lower_values(joint_count);
    Eigen::VectorXd upper_values(joint_count);
    std::vector<bool> pitch_values(joint_count, false);
    auto q_input = initial_q.unchecked<2>();
    auto lower_input = lower.unchecked<2>();
    auto upper_input = upper.unchecked<2>();
    auto pitch_input = pitch_mask.unchecked<2>();
    for (int module = 0; module < module_count_; ++module) {
      for (int local = 0; local < local_joint_count_; ++local) {
        const int index = module * local_joint_count_ + local;
        initial[index] = q_input(module, local);
        lower_values[index] = lower_input(module, local);
        upper_values[index] = upper_input(module, local);
        pitch_values[index] = pitch_input(module, local) != 0;
        initial[index] = std::clamp(
            initial[index], lower_values[index], upper_values[index]);
      }
    }

    std::vector<AnchorSpec> anchors;
    anchors.reserve(anchor_count);
    auto anchor_modules = anchor_module.unchecked<1>();
    auto anchor_links = anchor_link.unchecked<1>();
    auto local_rotations = anchor_local_r.unchecked<3>();
    auto local_translations = anchor_local_p.unchecked<2>();
    auto target_rotations = target_r.unchecked<3>();
    auto target_translations = target_p.unchecked<2>();
    for (py::ssize_t index = 0; index < anchor_count; ++index) {
      AnchorSpec anchor;
      anchor.module = anchor_modules(index);
      anchor.link = anchor_links(index);
      for (int row = 0; row < 3; ++row) {
        anchor.local.translation[row] = local_translations(index, row);
        anchor.target.translation[row] = target_translations(index, row);
        for (int column = 0; column < 3; ++column) {
          anchor.local.rotation(row, column) =
              local_rotations(index, row, column);
          anchor.target.rotation(row, column) =
              target_rotations(index, row, column);
        }
      }
      anchors.push_back(anchor);
    }

    const Transform centroidal{
        read_matrix(centroidal_r.data()), read_vector(centroidal_p.data())};
    SolverConfig config;
    config.maximum_iterations =
        py::cast<int>(config_values["maximum_iterations"]);
    config.relaxed_seed_maximum_iterations =
        py::cast<int>(config_values["relaxed_seed_maximum_iterations"]);
    config.feasible_refinement_iterations =
        py::cast<int>(config_values["feasible_refinement_iterations"]);
    config.damping = py::cast<double>(config_values["damping"]);
    config.position_weight =
        py::cast<double>(config_values["position_weight"]);
    config.attitude_weight =
        py::cast<double>(config_values["attitude_weight"]);
    config.continuity_regularization_weight = py::cast<double>(
        config_values["continuity_regularization_weight"]);
    config.pitch_joint_regularization_weight = py::cast<double>(
        config_values["pitch_joint_regularization_weight"]);
    config.finite_difference_step_rad =
        py::cast<double>(config_values["finite_difference_step_rad"]);
    config.maximum_joint_step_rad =
        py::cast<double>(config_values["maximum_joint_step_rad"]);
    config.maximum_base_translation_step_m = py::cast<double>(
        config_values["maximum_base_translation_step_m"]);
    config.maximum_base_rotation_step_rad = py::cast<double>(
        config_values["maximum_base_rotation_step_rad"]);
    config.anchor_position_tolerance_m =
        py::cast<double>(config_values["anchor_position_tolerance_m"]);
    config.anchor_attitude_tolerance_rad =
        py::cast<double>(config_values["anchor_attitude_tolerance_rad"]);
    if (config_values.contains("use_relaxed_seed")) {
      config.use_relaxed_seed =
          py::cast<bool>(config_values["use_relaxed_seed"]);
    }
    if (config_values.contains("collision_margin_m")) {
      config.collision_margin_m =
          py::cast<double>(config_values["collision_margin_m"]);
    }
    if (config_values.contains("collision_feasibility_tolerance_m")) {
      config.collision_feasibility_tolerance_m =
          py::cast<double>(
              config_values["collision_feasibility_tolerance_m"]);
    }
    if (config_values.contains("collision_activation_distance_m")) {
      config.collision_activation_distance_m = py::cast<double>(
          config_values["collision_activation_distance_m"]);
    }
    if (config_values.contains("collision_weight")) {
      config.collision_weight =
          py::cast<double>(config_values["collision_weight"]);
    }
    if (config_values.contains("collision_refinement_iterations")) {
      config.collision_refinement_iterations = py::cast<int>(
          config_values["collision_refinement_iterations"]);
    }
    if (config_values.contains("collision_line_search_steps")) {
      config.collision_line_search_steps = py::cast<int>(
          config_values["collision_line_search_steps"]);
    }

    const NativeSolution solution = solve_centroidal_impl(
        initial,
        lower_values,
        upper_values,
        pitch_values,
        centroidal,
        anchors,
        config);

    py::array_t<double> output_q(
        std::vector<py::ssize_t>{module_count_, local_joint_count_});
    auto q_output = output_q.mutable_unchecked<2>();
    for (int module = 0; module < module_count_; ++module) {
      for (int local = 0; local < local_joint_count_; ++local) {
        q_output(module, local) =
            solution.q[module * local_joint_count_ + local];
      }
    }
    py::array_t<double> output_base_r(
        std::vector<py::ssize_t>{3, 3});
    py::array_t<double> output_base_p(std::vector<py::ssize_t>{3});
    write_matrix(output_base_r.mutable_data(), solution.base.rotation);
    write_vector(output_base_p.mutable_data(), solution.base.translation);
    py::array_t<double> output_anchor_r(
        std::vector<py::ssize_t>{anchor_count, 3, 3});
    py::array_t<double> output_anchor_p(
        std::vector<py::ssize_t>{anchor_count, 3});
    auto anchor_r_output = output_anchor_r.mutable_unchecked<3>();
    auto anchor_p_output = output_anchor_p.mutable_unchecked<2>();
    for (py::ssize_t index = 0; index < anchor_count; ++index) {
      for (int row = 0; row < 3; ++row) {
        anchor_p_output(index, row) =
            solution.evaluation.anchors[index].translation[row];
        for (int column = 0; column < 3; ++column) {
          anchor_r_output(index, row, column) =
              solution.evaluation.anchors[index].rotation(row, column);
        }
      }
    }
    py::array_t<double> output_com(std::vector<py::ssize_t>{3});
    write_vector(output_com.mutable_data(), solution.evaluation.com);
    py::dict output;
    output["feasible"] = solution.feasible;
    output["q"] = output_q;
    output["base_r"] = output_base_r;
    output["base_p"] = output_base_p;
    output["anchor_r"] = output_anchor_r;
    output["anchor_p"] = output_anchor_p;
    output["com"] = output_com;
    output["maximum_position_error"] =
        solution.maximum_position_error;
    output["maximum_attitude_error"] =
        solution.maximum_attitude_error;
    output["iterations"] = solution.iterations;
    output["minimum_proxy_clearance"] =
        solution.minimum_proxy_clearance;
    output["active_proxy_pair_count"] =
        solution.active_proxy_pair_count;
    return output;
  }

  py::dict check_collisions(
      py::array_t<double, py::array::c_style | py::array::forcecast> q,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          centroidal_r,
      py::array_t<double, py::array::c_style | py::array::forcecast>
          centroidal_p,
      bool exact,
      double margin_m,
      double max_selected_contact_penetration_m,
      double ground_plane_z_m) const {
    require_rank(q, 2, "q");
    require_rank(centroidal_r, 2, "centroidal_r");
    require_rank(centroidal_p, 1, "centroidal_p");
    if (q.shape(0) != module_count_ ||
        q.shape(1) != local_joint_count_ ||
        centroidal_r.shape(0) != 3 || centroidal_r.shape(1) != 3 ||
        centroidal_p.shape(0) != 3) {
      throw std::invalid_argument(
          "collision checker input has invalid shape");
    }
    if (!active_collision_scene_.enabled) {
      throw std::runtime_error("collision scene is not configured");
    }
    if (!std::isfinite(max_selected_contact_penetration_m) ||
        max_selected_contact_penetration_m < 0.0) {
      throw std::invalid_argument(
          "maximum selected-contact penetration must be finite and "
          "non-negative");
    }
    if (!std::isfinite(ground_plane_z_m) &&
        !std::isnan(ground_plane_z_m)) {
      throw std::invalid_argument(
          "ground plane z must be finite or NaN when disabled");
    }
    Eigen::VectorXd q_values(module_count_ * local_joint_count_);
    auto input = q.unchecked<2>();
    for (int module = 0; module < module_count_; ++module) {
      for (int local = 0; local < local_joint_count_; ++local) {
        q_values[module * local_joint_count_ + local] =
            input(module, local);
      }
    }
    const Transform centroidal{
        read_matrix(centroidal_r.data()),
        read_vector(centroidal_p.data())};
    // Consecutive object/support checks share exactly the same robot state.
    // Preserve arithmetic and reports; reuse only bit-identical inputs. Scene
    // pose, contact permissions and ground checks are always applied afresh.
    const bool same_configuration = collision_configuration_valid_ &&
        collision_configuration_exact_ == exact &&
        collision_configuration_q_.size() == q_values.size() &&
        std::memcmp(collision_configuration_q_.data(), q_values.data(),
                    sizeof(double) * q_values.size()) == 0 &&
        std::memcmp(collision_configuration_pose_.rotation.data(), centroidal.rotation.data(),
                    sizeof(double) * 9) == 0 &&
        std::memcmp(collision_configuration_pose_.translation.data(), centroidal.translation.data(),
                    sizeof(double) * 3) == 0;
    if (!same_configuration) {
      collision_configuration_evaluation_ = evaluate_recentered(q_values, centroidal, {}).second;
      collision_configuration_frames_ = collision_frame_cache(
          collision_configuration_evaluation_, active_collision_scene_, exact);
      collision_configuration_q_ = q_values;
      collision_configuration_pose_ = centroidal;
      collision_configuration_exact_ = exact;
      collision_configuration_valid_ = true;
      collision_robot_metrics_.clear();
    }
    if (collision_configuration_margin_ != margin_m) {
      collision_robot_metrics_.clear();
      collision_configuration_margin_ = margin_m;
    }
    collision_configuration_frames_.object_aabb = box_world_aabb(
        active_collision_scene_.object_pose, 0.5 * active_collision_scene_.object_size);
    const Evaluation& evaluation = collision_configuration_evaluation_;
    const CollisionFrameCache& frames = collision_configuration_frames_;
    const CollisionMetrics metrics = collision_metrics(
        evaluation,
        active_collision_scene_,
        exact,
        margin_m, &frames, &collision_robot_metrics_);
    const bool ground_plane_enabled = std::isfinite(ground_plane_z_m);
    double minimum_ground_clearance_m =
        std::numeric_limits<double>::infinity();
    int ground_violating_proxy_count = 0;
    py::list ground_violating_proxies;
    if (ground_plane_enabled) {
      for (int instance = 0;
           instance < static_cast<int>(frames.proxy_aabbs.size());
           ++instance) {
        const double clearance =
            frames.proxy_aabbs[instance].first[2] - ground_plane_z_m;
        minimum_ground_clearance_m =
            std::min(minimum_ground_clearance_m, clearance);
        if (clearance + 1.0e-12 >= margin_m) {
          continue;
        }
        ++ground_violating_proxy_count;
        py::dict detail;
        detail["instance"] = instance;
        detail["clearance_m"] = clearance;
        ground_violating_proxies.append(std::move(detail));
      }
    }
    double maximum_selected_contact_penetration_m = 0.0;
    int selected_contact_penetration_violating_pair_count = 0;
    py::list selected_contact_pairs;
    for (const CollisionPair& pair :
         active_collision_scene_.selected_contact_pairs) {
      const double clearance = exact
          ? pair_clearance(evaluation, active_collision_scene_, pair, true)
          : cached_pair_clearance(active_collision_scene_, pair, false, frames);
      const double penetration = std::max(0.0, -clearance);
      const bool violating =
          penetration > max_selected_contact_penetration_m + 1.0e-12;
      maximum_selected_contact_penetration_m = std::max(
          maximum_selected_contact_penetration_m, penetration);
      if (violating) {
        ++selected_contact_penetration_violating_pair_count;
      }
      py::dict detail;
      detail["first_instance"] = pair.first_instance;
      detail["second_kind"] = "object";
      detail["clearance_m"] = clearance;
      detail["penetration_m"] = penetration;
      detail["violating"] = violating;
      selected_contact_pairs.append(std::move(detail));
    }
    py::dict output;
    output["minimum_clearance_m"] = metrics.minimum;
    double minimum_robot_clearance_m =
        std::numeric_limits<double>::infinity();
    double minimum_object_clearance_m =
        std::numeric_limits<double>::infinity();
    int robot_colliding_pair_count = 0;
    int object_colliding_pair_count = 0;
    for (int pair_index = 0;
         pair_index < static_cast<int>(metrics.clearances.size());
         ++pair_index) {
      const CollisionPair& pair =
          active_collision_scene_.pairs[pair_index];
      const bool object_pair =
          pair.second_kind == CollisionTargetKind::kObject;
      double& minimum = object_pair
          ? minimum_object_clearance_m
          : minimum_robot_clearance_m;
      minimum = std::min(minimum, metrics.clearances[pair_index]);
      if (metrics.colliding[pair_index]) {
        if (object_pair) {
          ++object_colliding_pair_count;
        } else {
          ++robot_colliding_pair_count;
        }
      }
    }
    output["minimum_robot_clearance_m"] =
        minimum_robot_clearance_m;
    output["minimum_object_clearance_m"] =
        minimum_object_clearance_m;
    output["robot_colliding_pair_count"] =
        robot_colliding_pair_count;
    output["object_colliding_pair_count"] =
        object_colliding_pair_count;
    output["violating_pair_count"] = metrics.active_count;
    output["colliding_pair_count"] = metrics.colliding_count;
    output["checked_pair_count"] =
        metrics.evaluated_count;
    output["narrow_phase_pair_count"] =
        metrics.narrow_phase_count;
    output["broad_phase_pruned_pair_count"] =
        metrics.broad_phase_pruned_count;
    output["accepted"] =
        metrics.colliding_count == 0 &&
        metrics.minimum + 1.0e-12 >= margin_m &&
        selected_contact_penetration_violating_pair_count == 0 &&
        ground_violating_proxy_count == 0;
    output["ground_plane_enabled"] = ground_plane_enabled;
    if (ground_plane_enabled) {
      output["ground_plane_z_m"] = ground_plane_z_m;
      output["minimum_ground_clearance_m"] =
          minimum_ground_clearance_m;
    } else {
      output["ground_plane_z_m"] = py::none();
      output["minimum_ground_clearance_m"] = py::none();
    }
    output["ground_violating_proxy_count"] =
        ground_violating_proxy_count;
    output["ground_violating_proxies"] =
        std::move(ground_violating_proxies);
    output["selected_contact_pair_count"] =
        active_collision_scene_.selected_contact_pairs.size();
    output["selected_contact_penetration_limit_m"] =
        max_selected_contact_penetration_m;
    output["maximum_selected_contact_penetration_m"] =
        maximum_selected_contact_penetration_m;
    output["selected_contact_penetration_violating_pair_count"] =
        selected_contact_penetration_violating_pair_count;
    output["selected_contact_pairs"] =
        std::move(selected_contact_pairs);
    // Only near/violating pairs can appear in the diagnostic list. Sorting
    // all O(body_count^2) distant pairs used to dominate this control query.
    std::vector<int> ordered;
    for (int index = 0;
         index < static_cast<int>(metrics.clearances.size());
         ++index) {
      if (metrics.colliding[index] || metrics.clearances[index] <= std::max(margin_m, 0.001))
        ordered.push_back(index);
    }
    std::sort(
        ordered.begin(),
        ordered.end(),
        [&metrics](int left, int right) {
          return metrics.clearances[left] <
                 metrics.clearances[right];
        });
    py::list worst_pairs;
    for (int pair_index : ordered) {
      const CollisionPair& pair =
          active_collision_scene_.pairs[pair_index];
      const bool colliding = metrics.colliding[pair_index];
      if (py::len(worst_pairs) >= 32) {
        break;
      }
      if (!colliding &&
          metrics.clearances[pair_index] >
              std::max(margin_m, 0.001)) {
        continue;
      }
      py::dict detail;
      detail["first_instance"] = pair.first_instance;
      detail["second_kind"] =
          pair.second_kind == CollisionTargetKind::kObject
              ? "object"
              : "robot";
      detail["second_instance"] = pair.second_instance;
      detail["clearance_m"] = metrics.clearances[pair_index];
      detail["colliding"] = colliding;
      worst_pairs.append(std::move(detail));
    }
    output["worst_pairs"] = std::move(worst_pairs);
    return output;
  }

 private:
  int collision_ancestor(int link) const {
    std::unordered_set<int> collision_links;
    for (const CollisionGeometry& geometry : collision_geometry_) {
      collision_links.insert(geometry.link);
    }
    int current = link;
    for (int depth = 0; depth <= link_count_; ++depth) {
      if (collision_links.count(current) != 0) {
        return current;
      }
      bool advanced = false;
      for (const Joint& joint : joints_) {
        if (joint.child_link == current) {
          current = joint.parent_link;
          advanced = true;
          break;
        }
      }
      if (!advanced) {
        return link;
      }
    }
    return link;
  }

  bool kinematically_adjacent(int first_link, int second_link) const {
    for (const Joint& joint : joints_) {
      if ((joint.parent_link == first_link &&
           joint.child_link == second_link) ||
          (joint.parent_link == second_link &&
           joint.child_link == first_link)) {
        return true;
      }
    }
    return false;
  }

  std::unordered_set<int> controlled_ancestor_joints(int link) const {
    std::unordered_set<int> ancestors;
    int current = link;
    for (int depth = 0; depth <= link_count_; ++depth) {
      bool advanced = false;
      for (const Joint& joint : joints_) {
        if (joint.child_link == current) {
          if (joint.q_index >= 0) {
            ancestors.insert(joint.q_index);
          }
          current = joint.parent_link;
          advanced = true;
          break;
        }
      }
      if (!advanced) {
        break;
      }
    }
    return ancestors;
  }

  bool relative_pose_depends_on_controlled_joint(
      int first_link, int second_link) const {
    const std::unordered_set<int> first =
        controlled_ancestor_joints(first_link);
    const std::unordered_set<int> second =
        controlled_ancestor_joints(second_link);
    for (int joint : first) {
      if (second.count(joint) == 0) {
        return true;
      }
    }
    for (int joint : second) {
      if (first.count(joint) == 0) {
        return true;
      }
    }
    return false;
  }

  bool intended_dock_pair(
      int first_module,
      int first_link,
      int second_module,
      int second_link) const {
    for (const Edge& edge : edges_) {
      const int parent_link = collision_ancestor(edge.parent_link);
      const int child_link = collision_ancestor(edge.child_link);
      if ((edge.parent_module == first_module &&
           edge.child_module == second_module &&
           parent_link == first_link &&
           child_link == second_link) ||
          (edge.parent_module == second_module &&
           edge.child_module == first_module &&
           parent_link == second_link &&
           child_link == first_link)) {
        return true;
      }
    }
    return false;
  }

  void build_collision_pairs(
      bool object_enabled,
      CollisionScene* scene) const {
    if (!exact_collision_geometry_loaded_ &&
        !nominal_collision_pair_table_complete_) {
      throw std::runtime_error(
          "proxy-only collision geometry requires a complete nominal "
          "collision pair table");
    }
    scene->pairs.clear();
    const int geometry_count =
        static_cast<int>(collision_geometry_.size());
    const int instance_count = module_count_ * geometry_count;
    const Eigen::VectorXd nominal_q =
        Eigen::VectorXd::Zero(module_count_ * local_joint_count_);
    const Evaluation nominal =
        evaluate_one(nominal_q, Transform{}, {});
    for (int first = 0; first < instance_count; ++first) {
      const int first_module = first / geometry_count;
      const int first_geometry = first % geometry_count;
      const int first_link =
          collision_geometry_[first_geometry].link;
      for (int second = first + 1;
           second < instance_count;
           ++second) {
        const int second_module = second / geometry_count;
        const int second_geometry = second % geometry_count;
        const int second_link =
            collision_geometry_[second_geometry].link;
        bool proxy_enabled = true;
        if (first_module == second_module) {
          if (first_link == second_link ||
              kinematically_adjacent(first_link, second_link)) {
            continue;
          }
          const int low =
              std::min(first_geometry, second_geometry);
          const int high =
              std::max(first_geometry, second_geometry);
          const std::uint64_t geometry_pair =
              instance_key(low, high);
          const auto known =
              nominal_same_module_collision_.find(geometry_pair);
          bool authored_overlap = false;
          if (known != nominal_same_module_collision_.end()) {
            authored_overlap = known->second;
          } else if (exact_collision_geometry_loaded_) {
            const CollisionPair nominal_pair{
                low,
                CollisionTargetKind::kRobot,
                high};
            authored_overlap = pair_collides(
                nominal, *scene, nominal_pair, true);
            nominal_same_module_collision_[geometry_pair] =
                authored_overlap;
          }
          if (authored_overlap) {
            continue;
          }
          proxy_enabled =
              relative_pose_depends_on_controlled_joint(
                  first_link, second_link);
        } else if (intended_dock_pair(
                       first_module,
                       first_link,
                       second_module,
                       second_link)) {
          continue;
        }
        scene->pairs.push_back(
            CollisionPair{
                first,
                CollisionTargetKind::kRobot,
                second,
                proxy_enabled});
      }
      if (object_enabled &&
          scene->allowed_object_instances.count(
              instance_key(first_module, first_geometry)) == 0) {
        scene->pairs.push_back(
            CollisionPair{
                first,
                CollisionTargetKind::kObject,
                -1});
      }
    }
    if (exact_collision_geometry_loaded_) {
      nominal_collision_pair_table_complete_ = true;
    }
  }

  Transform collision_geometry_pose(
      const Evaluation& evaluation,
      int instance,
      bool exact) const {
    const int geometry_count =
        static_cast<int>(collision_geometry_.size());
    const int module = instance / geometry_count;
    const int geometry_index = instance % geometry_count;
    const CollisionGeometry& geometry =
        collision_geometry_[geometry_index];
    const Transform& link_pose =
        evaluation.links[module * link_count_ + geometry.link];
    return compose(
        link_pose,
        exact ? geometry.exact_local : geometry.proxy_local);
  }

  CollisionFrameCache collision_frame_cache(
      const Evaluation& evaluation,
      const CollisionScene& scene,
      bool exact) const {
    const int geometry_count =
        static_cast<int>(collision_geometry_.size());
    const int instance_count = module_count_ * geometry_count;
    CollisionFrameCache cache;
    cache.proxy_poses.reserve(instance_count);
    cache.proxy_aabbs.reserve(instance_count);
    if (exact) {
      cache.exact_poses.reserve(instance_count);
    }
    for (int instance = 0; instance < instance_count; ++instance) {
      const int geometry_index = instance % geometry_count;
      const Transform proxy_pose =
          collision_geometry_pose(evaluation, instance, false);
      cache.proxy_poses.push_back(proxy_pose);
      cache.proxy_aabbs.push_back(box_world_aabb(
          proxy_pose,
          collision_geometry_[geometry_index].half_extent));
      if (exact) {
        cache.exact_poses.push_back(
            collision_geometry_pose(evaluation, instance, true));
      }
    }
    cache.object_aabb = box_world_aabb(
        scene.object_pose, 0.5 * scene.object_size);
    return cache;
  }

  double cached_pair_clearance(
      const CollisionScene& scene,
      const CollisionPair& pair,
      bool exact,
      const CollisionFrameCache& frames) const {
    const int geometry_count =
        static_cast<int>(collision_geometry_.size());
    const int first_geometry = pair.first_instance % geometry_count;
    const CollisionGeometry& first =
        collision_geometry_[first_geometry];
    const Transform& first_pose =
        exact
            ? frames.exact_poses[pair.first_instance]
            : frames.proxy_poses[pair.first_instance];
    const std::shared_ptr<fcl::CollisionGeometryd> first_shape =
        exact
            ? std::static_pointer_cast<fcl::CollisionGeometryd>(
                  first.exact)
            : std::static_pointer_cast<fcl::CollisionGeometryd>(
                  first.proxy);
    if (!first_shape) {
      throw std::runtime_error(
          "exact collision geometry was not loaded");
    }
    if (pair.second_kind == CollisionTargetKind::kObject) {
      if (!scene.object_shape) {
        throw std::runtime_error(
            "object collision pair lacks a cached object shape");
      }
      return fcl_distance(
          first_shape,
          first_pose,
          std::static_pointer_cast<fcl::CollisionGeometryd>(
              scene.object_shape),
          scene.object_pose);
    }
    const int second_geometry =
        pair.second_instance % geometry_count;
    const CollisionGeometry& second =
        collision_geometry_[second_geometry];
    const Transform& second_pose =
        exact
            ? frames.exact_poses[pair.second_instance]
            : frames.proxy_poses[pair.second_instance];
    const std::shared_ptr<fcl::CollisionGeometryd> second_shape =
        exact
            ? std::static_pointer_cast<fcl::CollisionGeometryd>(
                  second.exact)
            : std::static_pointer_cast<fcl::CollisionGeometryd>(
                  second.proxy);
    if (!second_shape) {
      throw std::runtime_error(
          "exact collision geometry was not loaded");
    }
    return fcl_distance(
        first_shape, first_pose, second_shape, second_pose);
  }

  double cached_pair_aabb_clearance(
      const CollisionPair& pair,
      const CollisionFrameCache& frames) const {
    return aabb_clearance(
        frames.proxy_aabbs[pair.first_instance],
        pair.second_kind == CollisionTargetKind::kObject
            ? frames.object_aabb
            : frames.proxy_aabbs[pair.second_instance]);
  }

  double pair_clearance(
      const Evaluation& evaluation,
      const CollisionScene& scene,
      const CollisionPair& pair,
      bool exact) const {
    const int geometry_count =
        static_cast<int>(collision_geometry_.size());
    const int first_geometry = pair.first_instance % geometry_count;
    const CollisionGeometry& first =
        collision_geometry_[first_geometry];
    const Transform first_pose = collision_geometry_pose(
        evaluation, pair.first_instance, exact);
    const std::shared_ptr<fcl::CollisionGeometryd> first_shape =
        exact
            ? std::static_pointer_cast<fcl::CollisionGeometryd>(
                  first.exact)
            : std::static_pointer_cast<fcl::CollisionGeometryd>(
                  first.proxy);
    if (!first_shape) {
      throw std::runtime_error(
          "exact collision geometry was not loaded");
    }
    if (pair.second_kind == CollisionTargetKind::kObject) {
      if (exact) {
        const Transform first_proxy_pose =
            collision_geometry_pose(
                evaluation, pair.first_instance, false);
        const double broad_phase_clearance = aabb_clearance(
            box_world_aabb(
                first_proxy_pose, first.half_extent),
            box_world_aabb(
                scene.object_pose, 0.5 * scene.object_size));
        if (broad_phase_clearance > 0.0) {
          return broad_phase_clearance;
        }
      }
      if (!scene.object_shape) {
        throw std::runtime_error(
            "object collision pair lacks a cached object shape");
      }
      return fcl_distance(
          first_shape,
          first_pose,
          std::static_pointer_cast<fcl::CollisionGeometryd>(
              scene.object_shape),
          scene.object_pose);
    }
    const int second_geometry = pair.second_instance % geometry_count;
    const CollisionGeometry& second =
        collision_geometry_[second_geometry];
    const Transform second_pose = collision_geometry_pose(
        evaluation, pair.second_instance, exact);
    if (exact) {
      const Transform first_proxy_pose =
          collision_geometry_pose(
              evaluation, pair.first_instance, false);
      const Transform second_proxy_pose =
          collision_geometry_pose(
              evaluation, pair.second_instance, false);
      const double broad_phase_clearance = aabb_clearance(
          box_world_aabb(
              first_proxy_pose, first.half_extent),
          box_world_aabb(
              second_proxy_pose, second.half_extent));
      if (broad_phase_clearance > 0.0) {
        return broad_phase_clearance;
      }
    }
    const std::shared_ptr<fcl::CollisionGeometryd> second_shape =
        exact
            ? std::static_pointer_cast<fcl::CollisionGeometryd>(
                  second.exact)
            : std::static_pointer_cast<fcl::CollisionGeometryd>(
                  second.proxy);
    if (!second_shape) {
      throw std::runtime_error(
          "exact collision geometry was not loaded");
    }
    return fcl_distance(
        first_shape, first_pose, second_shape, second_pose);
  }

  bool pair_collides(
      const Evaluation& evaluation,
      const CollisionScene& scene,
      const CollisionPair& pair,
      bool exact) const {
    const int geometry_count =
        static_cast<int>(collision_geometry_.size());
    const int first_geometry = pair.first_instance % geometry_count;
    const CollisionGeometry& first =
        collision_geometry_[first_geometry];
    const Transform first_pose = collision_geometry_pose(
        evaluation, pair.first_instance, exact);
    const std::shared_ptr<fcl::CollisionGeometryd> first_shape =
        exact
            ? std::static_pointer_cast<fcl::CollisionGeometryd>(
                  first.exact)
            : std::static_pointer_cast<fcl::CollisionGeometryd>(
                  first.proxy);
    if (!first_shape) {
      throw std::runtime_error(
          "exact collision geometry was not loaded");
    }
    if (pair.second_kind == CollisionTargetKind::kObject) {
      if (!scene.object_shape) {
        throw std::runtime_error(
            "object collision pair lacks a cached object shape");
      }
      return fcl_collides(
          first_shape,
          first_pose,
          std::static_pointer_cast<fcl::CollisionGeometryd>(
              scene.object_shape),
          scene.object_pose);
    }
    const int second_geometry = pair.second_instance % geometry_count;
    const CollisionGeometry& second =
        collision_geometry_[second_geometry];
    const Transform second_pose = collision_geometry_pose(
        evaluation, pair.second_instance, exact);
    const std::shared_ptr<fcl::CollisionGeometryd> second_shape =
        exact
            ? std::static_pointer_cast<fcl::CollisionGeometryd>(
                  second.exact)
            : std::static_pointer_cast<fcl::CollisionGeometryd>(
                  second.proxy);
    if (!second_shape) {
      throw std::runtime_error(
          "exact collision geometry was not loaded");
    }
    return fcl_collides(
        first_shape, first_pose, second_shape, second_pose);
  }

  CollisionMetrics collision_metrics(
      const Evaluation& evaluation,
      const CollisionScene& scene,
      bool exact,
      double activation_distance,
      const CollisionFrameCache* prepared_frames = nullptr,
      std::vector<CachedRobotPairMetric>* robot_cache = nullptr) const {
    CollisionMetrics output;
    output.clearances.reserve(scene.pairs.size());
    output.colliding.reserve(scene.pairs.size());
    const CollisionFrameCache local_frames = prepared_frames == nullptr
        ? collision_frame_cache(evaluation, scene, exact) : CollisionFrameCache{};
    const CollisionFrameCache& frames = prepared_frames == nullptr ? local_frames : *prepared_frames;
    std::size_t robot_index = 0;
    for (const CollisionPair& pair : scene.pairs) {
      if (!exact && !pair.proxy_enabled) {
        output.clearances.push_back(
            std::numeric_limits<double>::infinity());
        output.colliding.push_back(false);
        continue;
      }
      ++output.evaluated_count;
      const bool robot_pair = pair.second_kind == CollisionTargetKind::kRobot;
      const bool reuse = robot_pair && robot_cache != nullptr && robot_index < robot_cache->size();
      const double broad_phase_clearance = reuse ? 0.0 :
          cached_pair_aabb_clearance(pair, frames);
      const bool broad_phase_safe = reuse ? (*robot_cache)[robot_index].broad_phase_safe :
          exact
              ? broad_phase_clearance > 0.0
              : broad_phase_clearance > activation_distance;
      const double clearance = reuse ? (*robot_cache)[robot_index].clearance :
          broad_phase_safe
              ? broad_phase_clearance
              : cached_pair_clearance(
                    scene, pair, exact, frames);
      if (broad_phase_safe) {
        ++output.broad_phase_pruned_count;
      } else {
        ++output.narrow_phase_count;
      }
      output.clearances.push_back(clearance);
      const bool colliding = reuse ? (*robot_cache)[robot_index].colliding :
          exact && clearance <= 0.0 &&
          pair_collides(evaluation, scene, pair, true);
      if (robot_pair && robot_cache != nullptr) {
        if (!reuse) robot_cache->push_back({clearance, colliding, broad_phase_safe});
        ++robot_index;
      }
      output.colliding.push_back(colliding);
      output.minimum = std::min(output.minimum, clearance);
      if (clearance < activation_distance) {
        ++output.active_count;
      }
      if (colliding) {
        ++output.colliding_count;
      }
    }
    return output;
  }

  std::vector<Transform> evaluate_local_links(const Eigen::VectorXd& q) const {
    const std::size_t local_count =
        static_cast<std::size_t>(module_count_ * link_count_);
    std::vector<Transform> local(local_count);
    std::vector<Transform> link_root(link_count_);
    for (int module = 0; module < module_count_; ++module) {
      std::fill(link_root.begin(), link_root.end(), Transform{});
      link_root[root_link_] = Transform{};
      for (const Joint& joint : joints_) {
        Transform motion;
        const double value =
            joint.q_index < 0
                ? 0.0
                : q[module * local_joint_count_ + joint.q_index];
        if (joint.type == 1) {
          motion.rotation =
              Eigen::AngleAxisd(value, joint.axis).toRotationMatrix();
        } else if (joint.type == 2) {
          motion.translation = value * joint.axis;
        }
        link_root[joint.child_link] = compose(
            compose(link_root[joint.parent_link], joint.origin), motion);
      }
      const Transform root_from_base = inverse(link_root[base_link_]);
      for (int link = 0; link < link_count_; ++link) {
        local[module * link_count_ + link] =
            compose(root_from_base, link_root[link]);
      }
    }

    return local;
  }

  Evaluation evaluate_one(
      const Eigen::VectorXd& q,
      const Transform& base,
      const std::vector<AnchorSpec>& anchors,
      const std::vector<Transform>* prepared_local = nullptr) const {
    const std::vector<Transform> computed_local = prepared_local == nullptr
        ? evaluate_local_links(q) : std::vector<Transform>{};
    const std::vector<Transform>& local = prepared_local == nullptr
        ? computed_local : *prepared_local;
    const std::size_t local_count = local.size();
    Evaluation output;
    output.roots.assign(module_count_, Transform{});
    output.roots[base_module_] = base;
    for (const Edge& edge : edges_) {
      const Transform parent_connect = compose(
          output.roots[edge.parent_module],
          local[edge.parent_module * link_count_ + edge.parent_link]);
      output.roots[edge.child_module] = compose(
          compose(parent_connect, edge.relation),
          inverse(
              local[edge.child_module * link_count_ + edge.child_link]));
    }

    Vector3 weighted_com = Vector3::Zero();
    output.links.resize(local_count);
    for (int module = 0; module < module_count_; ++module) {
      for (int link = 0; link < link_count_; ++link) {
        const Transform world = compose(
            output.roots[module],
            local[module * link_count_ + link]);
        output.links[module * link_count_ + link] = world;
        weighted_com += link_masses_[link] *
                        (world.translation +
                         world.rotation * link_centers_[link]);
      }
    }
    output.com = weighted_com / (module_mass_ * module_count_);
    output.anchors.reserve(anchors.size());
    for (const AnchorSpec& anchor : anchors) {
      output.anchors.push_back(compose(
          compose(
              output.roots[anchor.module],
              local[anchor.module * link_count_ + anchor.link]),
          anchor.local));
    }
    return output;
  }

  std::pair<Transform, Evaluation> evaluate_recentered(
      const Eigen::VectorXd& q,
      const Transform& centroidal,
      const std::vector<AnchorSpec>& anchors) const {
    Transform zero_base;
    zero_base.rotation = centroidal.rotation;
    // Recentring changes only the world root pose, never local joint geometry.
    // Keep both world evaluations and all arithmetic in their original order.
    const std::vector<Transform> local = evaluate_local_links(q);
    Evaluation zero = evaluate_one(q, zero_base, anchors, &local);
    Transform base;
    base.rotation = centroidal.rotation;
    base.translation = centroidal.translation - zero.com;
    return {base, evaluate_one(q, base, anchors, &local)};
  }

  static Vector3 attitude_error(
      const Transform& target, const Transform& actual) {
    return rotation_log(
        target.rotation * actual.rotation.transpose());
  }

  static void maximum_errors(
      const Evaluation& evaluation,
      const std::vector<AnchorSpec>& anchors,
      double* maximum_position,
      double* maximum_attitude) {
    *maximum_position = 0.0;
    *maximum_attitude = 0.0;
    for (std::size_t index = 0; index < anchors.size(); ++index) {
      *maximum_position = std::max(
          *maximum_position,
          (anchors[index].target.translation -
           evaluation.anchors[index].translation)
              .norm());
      *maximum_attitude = std::max(
          *maximum_attitude,
          attitude_error(
              anchors[index].target, evaluation.anchors[index]).norm());
    }
  }

  Eigen::MatrixXd fixed_base_jacobian(
      const Eigen::VectorXd& q,
      const Eigen::VectorXd& lower,
      const Eigen::VectorXd& upper,
      const Transform& base,
      const std::vector<AnchorSpec>& anchors,
      const Evaluation& nominal,
      double requested_step) const {
    const int joint_count = q.size();
    const int row_count = static_cast<int>(anchors.size()) * 6;
    Eigen::MatrixXd jacobian(row_count, joint_count);
    for (int joint = 0; joint < joint_count; ++joint) {
      const double minus_room = q[joint] - lower[joint];
      const double plus_room = upper[joint] - q[joint];
      Eigen::VectorXd before_q = q;
      Eigen::VectorXd after_q = q;
      const Evaluation* before = &nominal;
      const Evaluation* after = &nominal;
      Evaluation before_value;
      Evaluation after_value;
      double denominator = 0.0;
      if (minus_room >= requested_step &&
          plus_room >= requested_step) {
        before_q[joint] -= requested_step;
        after_q[joint] += requested_step;
        before_value = evaluate_one(before_q, base, anchors);
        after_value = evaluate_one(after_q, base, anchors);
        before = &before_value;
        after = &after_value;
        denominator = 2.0 * requested_step;
      } else if (plus_room >= requested_step) {
        after_q[joint] += requested_step;
        after_value = evaluate_one(after_q, base, anchors);
        after = &after_value;
        denominator = requested_step;
      } else if (minus_room >= requested_step) {
        before_q[joint] -= requested_step;
        before_value = evaluate_one(before_q, base, anchors);
        before = &before_value;
        denominator = requested_step;
      } else if (plus_room > 1.0e-12) {
        const double step = std::min(requested_step, plus_room);
        after_q[joint] += step;
        after_value = evaluate_one(after_q, base, anchors);
        after = &after_value;
        denominator = step;
      } else if (minus_room > 1.0e-12) {
        const double step = std::min(requested_step, minus_room);
        before_q[joint] -= step;
        before_value = evaluate_one(before_q, base, anchors);
        before = &before_value;
        denominator = step;
      } else {
        throw std::runtime_error(
            "joint has no finite-difference room");
      }
      for (std::size_t anchor = 0; anchor < anchors.size(); ++anchor) {
        jacobian.block<3, 1>(anchor * 6, joint) =
            (after->anchors[anchor].translation -
             before->anchors[anchor].translation) /
            denominator;
        jacobian.block<3, 1>(anchor * 6 + 3, joint) =
            rotation_log(
                after->anchors[anchor].rotation *
                before->anchors[anchor].rotation.transpose()) /
            denominator;
      }
    }
    return jacobian;
  }

  Eigen::MatrixXd fixed_centroidal_jacobian(
      const Eigen::VectorXd& q,
      const Eigen::VectorXd& lower,
      const Eigen::VectorXd& upper,
      const Transform& centroidal,
      const std::vector<AnchorSpec>& anchors,
      const Evaluation& nominal,
      double requested_step) const {
    const int joint_count = q.size();
    Eigen::MatrixXd jacobian(
        static_cast<int>(anchors.size()) * 6, joint_count);
    for (int joint = 0; joint < joint_count; ++joint) {
      const double plus_room = upper[joint] - q[joint];
      const double minus_room = q[joint] - lower[joint];
      Eigen::VectorXd perturbed_q = q;
      bool forward = true;
      double denominator = 0.0;
      if (plus_room > 1.0e-12) {
        denominator = std::min(requested_step, plus_room);
        perturbed_q[joint] += denominator;
      } else if (minus_room > 1.0e-12) {
        denominator = std::min(requested_step, minus_room);
        perturbed_q[joint] -= denominator;
        forward = false;
      } else {
        throw std::runtime_error(
            "joint has no finite-difference room");
      }
      const Evaluation perturbed =
          evaluate_recentered(perturbed_q, centroidal, anchors).second;
      for (std::size_t anchor = 0; anchor < anchors.size(); ++anchor) {
        const Transform& before =
            forward ? nominal.anchors[anchor] : perturbed.anchors[anchor];
        const Transform& after =
            forward ? perturbed.anchors[anchor] : nominal.anchors[anchor];
        jacobian.block<3, 1>(anchor * 6, joint) =
            (after.translation - before.translation) / denominator;
        jacobian.block<3, 1>(anchor * 6 + 3, joint) =
            rotation_log(
                after.rotation * before.rotation.transpose()) /
            denominator;
      }
    }
    return jacobian;
  }

  Eigen::MatrixXd fixed_centroidal_collision_jacobian(
      const Eigen::VectorXd& q,
      const Eigen::VectorXd& lower,
      const Eigen::VectorXd& upper,
      const Transform& centroidal,
      const std::vector<AnchorSpec>& anchors,
      const CollisionScene& scene,
      const CollisionMetrics& nominal_metrics,
      const std::vector<int>& active_pairs,
      double requested_step) const {
    const int joint_count = q.size();
    Eigen::MatrixXd jacobian(
        static_cast<int>(active_pairs.size()), joint_count);
    for (int joint = 0; joint < joint_count; ++joint) {
      const double plus_room = upper[joint] - q[joint];
      const double minus_room = q[joint] - lower[joint];
      Eigen::VectorXd perturbed_q = q;
      bool forward = true;
      double denominator = 0.0;
      if (plus_room > 1.0e-12) {
        denominator = std::min(requested_step, plus_room);
        perturbed_q[joint] += denominator;
      } else if (minus_room > 1.0e-12) {
        denominator = std::min(requested_step, minus_room);
        perturbed_q[joint] -= denominator;
        forward = false;
      } else {
        throw std::runtime_error(
            "joint has no finite-difference room");
      }
      const Evaluation perturbed =
          evaluate_recentered(perturbed_q, centroidal, anchors).second;
      for (int row = 0;
           row < static_cast<int>(active_pairs.size());
           ++row) {
        const int pair_index = active_pairs[row];
        const double nominal =
            nominal_metrics.clearances[pair_index];
        const double changed = pair_clearance(
            perturbed,
            scene,
            scene.pairs[pair_index],
            false);
        jacobian(row, joint) =
            (forward ? changed - nominal : nominal - changed) /
            denominator;
      }
    }
    return jacobian;
  }

  static Eigen::VectorXd solve_normal(
      const Eigen::MatrixXd& matrix,
      const Eigen::VectorXd& residual,
      double damping,
      const Eigen::VectorXd& q,
      const Eigen::VectorXd& reference_q,
      const std::vector<bool>& pitch,
      double continuity_regularization,
      double pitch_regularization) {
    Eigen::MatrixXd normal = matrix.transpose() * matrix;
    normal.diagonal().array() += damping * damping;
    Eigen::VectorXd right = matrix.transpose() * residual;
    for (int joint = 0; joint < q.size(); ++joint) {
      normal(joint, joint) += continuity_regularization;
      right[joint] -=
          continuity_regularization * (q[joint] - reference_q[joint]);
      if (pitch[joint]) {
        normal(joint, joint) += pitch_regularization;
        right[joint] -= pitch_regularization * q[joint];
      }
    }
    Eigen::VectorXd result = normal.partialPivLu().solve(right);
    if (!result.allFinite()) {
      result = normal.completeOrthogonalDecomposition().solve(right);
    }
    return result;
  }

  std::pair<Eigen::VectorXd, int> relaxed_seed(
      const Eigen::VectorXd& initial_q,
      const Eigen::VectorXd& reference_q,
      const Eigen::VectorXd& lower,
      const Eigen::VectorXd& upper,
      const std::vector<bool>& pitch,
      const Transform& centroidal,
      const std::vector<AnchorSpec>& anchors,
      const SolverConfig& config) const {
    if (anchors.empty()) {
      return {initial_q, 0};
    }
    Eigen::VectorXd q = initial_q;
    Transform base = evaluate_recentered(q, centroidal, anchors).first;
    Eigen::VectorXd best_q = q;
    double best_objective = std::numeric_limits<double>::infinity();
    const int joint_count = q.size();
    const int row_count = static_cast<int>(anchors.size()) * 6;
    for (int iteration = 1;
         iteration <= config.relaxed_seed_maximum_iterations;
         ++iteration) {
      const Evaluation nominal = evaluate_one(q, base, anchors);
      const Eigen::MatrixXd joint_jacobian = fixed_base_jacobian(
          q,
          lower,
          upper,
          base,
          anchors,
          nominal,
          config.finite_difference_step_rad);
      Eigen::MatrixXd matrix(row_count, joint_count + 6);
      Eigen::VectorXd residual(row_count);
      double maximum_position = 0.0;
      double maximum_attitude = 0.0;
      for (std::size_t anchor = 0; anchor < anchors.size(); ++anchor) {
        const int row = static_cast<int>(anchor) * 6;
        const Vector3 position_error =
            anchors[anchor].target.translation -
            nominal.anchors[anchor].translation;
        const Vector3 rotation_error =
            attitude_error(anchors[anchor].target, nominal.anchors[anchor]);
        maximum_position =
            std::max(maximum_position, position_error.norm());
        maximum_attitude =
            std::max(maximum_attitude, rotation_error.norm());
        matrix.block(row, 0, 3, joint_count) =
            config.position_weight *
            joint_jacobian.block(row, 0, 3, joint_count);
        const Vector3 relative =
            nominal.anchors[anchor].translation - base.translation;
        Matrix3 base_position;
        base_position << 0.0, relative.z(), -relative.y(),
            -relative.z(), 0.0, relative.x(),
            relative.y(), -relative.x(), 0.0;
        matrix.block<3, 3>(row, joint_count) =
            config.position_weight * base_position;
        matrix.block<3, 3>(row, joint_count + 3) =
            config.position_weight * Matrix3::Identity();
        residual.segment<3>(row) =
            config.position_weight * position_error;

        matrix.block(row + 3, 0, 3, joint_count) =
            config.attitude_weight *
            joint_jacobian.block(row + 3, 0, 3, joint_count);
        matrix.block<3, 3>(row + 3, joint_count) =
            config.attitude_weight * Matrix3::Identity();
        matrix.block<3, 3>(row + 3, joint_count + 3).setZero();
        residual.segment<3>(row + 3) =
            config.attitude_weight * rotation_error;
      }
      const double objective = residual.squaredNorm();
      if (objective < best_objective) {
        best_objective = objective;
        best_q = q;
      }
      if (maximum_position <= config.anchor_position_tolerance_m &&
          maximum_attitude <= config.anchor_attitude_tolerance_rad) {
        return {q, iteration};
      }

      Eigen::MatrixXd normal = matrix.transpose() * matrix;
      normal.diagonal().array() += config.damping * config.damping;
      Eigen::VectorXd right = matrix.transpose() * residual;
      for (int joint = 0; joint < joint_count; ++joint) {
        normal(joint, joint) +=
            config.continuity_regularization_weight;
        right[joint] -= config.continuity_regularization_weight *
                        (q[joint] - reference_q[joint]);
        if (pitch[joint]) {
          normal(joint, joint) +=
              config.pitch_joint_regularization_weight;
          right[joint] -=
              config.pitch_joint_regularization_weight * q[joint];
        }
      }
      Eigen::VectorXd delta = normal.partialPivLu().solve(right);
      if (!delta.allFinite()) {
        delta = normal.completeOrthogonalDecomposition().solve(right);
      }
      for (int joint = 0; joint < joint_count; ++joint) {
        const double joint_delta = std::clamp(
            delta[joint],
            -config.maximum_joint_step_rad,
            config.maximum_joint_step_rad);
        q[joint] = std::clamp(
            q[joint] + joint_delta, lower[joint], upper[joint]);
      }
      const Vector3 rotation_delta = bounded_vector(
          delta.segment<3>(joint_count),
          config.maximum_base_rotation_step_rad);
      const Vector3 translation_delta = bounded_vector(
          delta.segment<3>(joint_count + 3),
          config.maximum_base_translation_step_m);
      base.rotation = rotation_from_vector(rotation_delta) * base.rotation;
      base.translation += translation_delta;
    }
    return {best_q, config.relaxed_seed_maximum_iterations};
  }

  NativeSolution solve_centroidal_impl(
      const Eigen::VectorXd& initial_q,
      const Eigen::VectorXd& lower,
      const Eigen::VectorXd& upper,
      const std::vector<bool>& pitch,
      const Transform& centroidal,
      const std::vector<AnchorSpec>& anchors,
      const SolverConfig& config) const {
    NativeSolution initial;
    initial.q = initial_q;
    auto initial_pair =
        evaluate_recentered(initial_q, centroidal, anchors);
    initial.base = initial_pair.first;
    initial.evaluation = std::move(initial_pair.second);
    maximum_errors(
        initial.evaluation,
        anchors,
        &initial.maximum_position_error,
        &initial.maximum_attitude_error);
    if (active_collision_scene_.enabled) {
      const CollisionMetrics metrics = collision_metrics(
          initial.evaluation,
          active_collision_scene_,
          false,
          config.collision_activation_distance_m);
      initial.minimum_proxy_clearance = metrics.minimum;
      initial.active_proxy_pair_count = metrics.active_count;
    }
    initial.feasible =
        initial.maximum_position_error <=
            config.anchor_position_tolerance_m &&
        initial.maximum_attitude_error <=
            config.anchor_attitude_tolerance_rad &&
        (!active_collision_scene_.enabled ||
         initial.minimum_proxy_clearance >=
             config.collision_margin_m -
                 config.collision_feasibility_tolerance_m);
    if (initial.feasible ||
        (anchors.empty() && !active_collision_scene_.enabled)) {
      return initial;
    }

    std::pair<Eigen::VectorXd, int> relaxed{initial_q, 0};
    if (config.use_relaxed_seed &&
        (initial.maximum_position_error > 0.15 ||
         initial.maximum_attitude_error > 0.50)) {
      relaxed = relaxed_seed(
          initial_q,
          initial_q,
          lower,
          upper,
          pitch,
          centroidal,
          anchors,
          config);
    }
    Eigen::VectorXd q = relaxed.first;
    const int relaxed_iterations = relaxed.second;
    NativeSolution best;
    double best_objective = std::numeric_limits<double>::infinity();
    int first_feasible_iteration = -1;
    NativeSolution last;
    last.q = q;
    last.base = centroidal;
    last.evaluation.anchors.resize(anchors.size());
    last.evaluation.com = centroidal.translation;
    last.maximum_position_error =
        std::numeric_limits<double>::infinity();
    last.maximum_attitude_error =
        std::numeric_limits<double>::infinity();
    const int joint_count = q.size();
    const int anchor_row_count =
        static_cast<int>(anchors.size()) * 6;
    const int maximum_iterations =
        config.maximum_iterations +
        (active_collision_scene_.enabled
             ? config.collision_refinement_iterations
             : 0);

    for (int iteration = 1;
         iteration <= maximum_iterations;
         ++iteration) {
      auto nominal_pair = evaluate_recentered(q, centroidal, anchors);
      const Transform& base = nominal_pair.first;
      const Evaluation& nominal = nominal_pair.second;
      const Eigen::MatrixXd jacobian = fixed_centroidal_jacobian(
          q,
          lower,
          upper,
          centroidal,
          anchors,
          nominal,
          config.finite_difference_step_rad);
      CollisionMetrics collision;
      std::vector<int> active_pairs;
      if (active_collision_scene_.enabled) {
        collision = collision_metrics(
            nominal,
            active_collision_scene_,
            false,
            config.collision_activation_distance_m);
        for (int pair = 0;
             pair < static_cast<int>(collision.clearances.size());
             ++pair) {
          if (collision.clearances[pair] <
              config.collision_activation_distance_m) {
            active_pairs.push_back(pair);
          }
        }
      }
      const int row_count =
          anchor_row_count + static_cast<int>(active_pairs.size());
      Eigen::MatrixXd matrix =
          Eigen::MatrixXd::Zero(row_count, joint_count);
      Eigen::VectorXd residual =
          Eigen::VectorXd::Zero(row_count);
      double maximum_position = 0.0;
      double maximum_attitude = 0.0;
      for (std::size_t anchor = 0; anchor < anchors.size(); ++anchor) {
        const int row = static_cast<int>(anchor) * 6;
        const Vector3 position_error =
            anchors[anchor].target.translation -
            nominal.anchors[anchor].translation;
        const Vector3 rotation_error =
            attitude_error(anchors[anchor].target, nominal.anchors[anchor]);
        maximum_position =
            std::max(maximum_position, position_error.norm());
        maximum_attitude =
            std::max(maximum_attitude, rotation_error.norm());
        matrix.block(row, 0, 3, joint_count) =
            config.position_weight *
            jacobian.block(row, 0, 3, joint_count);
        matrix.block(row + 3, 0, 3, joint_count) =
            config.attitude_weight *
            jacobian.block(row + 3, 0, 3, joint_count);
        residual.segment<3>(row) =
            config.position_weight * position_error;
        residual.segment<3>(row + 3) =
            config.attitude_weight * rotation_error;
      }
      if (!active_pairs.empty()) {
        const Eigen::MatrixXd collision_jacobian =
            fixed_centroidal_collision_jacobian(
                q,
                lower,
                upper,
                centroidal,
                anchors,
                active_collision_scene_,
                collision,
                active_pairs,
                config.finite_difference_step_rad);
        const double collision_scale =
            std::sqrt(config.collision_weight);
        for (int row = 0;
             row < static_cast<int>(active_pairs.size());
             ++row) {
          const int output_row = anchor_row_count + row;
          matrix.row(output_row) =
              collision_scale * collision_jacobian.row(row);
          residual[output_row] =
              collision_scale *
              std::max(
                  0.0,
                  config.collision_margin_m -
                      collision.clearances[active_pairs[row]]);
        }
      }
      double objective = residual.squaredNorm();
      for (int joint = 0; joint < joint_count; ++joint) {
        const double difference = q[joint] - initial_q[joint];
        objective += config.continuity_regularization_weight *
                     difference * difference;
        if (pitch[joint]) {
          objective += config.pitch_joint_regularization_weight *
                       q[joint] * q[joint];
        }
      }
      last.feasible = false;
      last.q = q;
      last.base = base;
      last.evaluation = nominal;
      last.maximum_position_error = maximum_position;
      last.maximum_attitude_error = maximum_attitude;
      last.minimum_proxy_clearance =
          active_collision_scene_.enabled
              ? collision.minimum
              : std::numeric_limits<double>::infinity();
      last.active_proxy_pair_count =
          static_cast<int>(active_pairs.size());
      last.iterations = relaxed_iterations + iteration;

      if (maximum_position <= config.anchor_position_tolerance_m &&
          maximum_attitude <= config.anchor_attitude_tolerance_rad &&
          (!active_collision_scene_.enabled ||
           collision.minimum >=
               config.collision_margin_m -
                   config.collision_feasibility_tolerance_m)) {
        if (first_feasible_iteration < 0) {
          first_feasible_iteration = iteration;
        }
        if (objective < best_objective) {
          best_objective = objective;
          best = last;
          best.feasible = true;
        }
        if (iteration - first_feasible_iteration >=
            config.feasible_refinement_iterations) {
          break;
        }
      }

      Eigen::VectorXd delta = solve_normal(
          matrix,
          residual,
          config.damping,
          q,
          initial_q,
          pitch,
          config.continuity_regularization_weight,
          config.pitch_joint_regularization_weight);
      for (int joint = 0; joint < joint_count; ++joint) {
        delta[joint] = std::clamp(
            delta[joint],
            -config.maximum_joint_step_rad,
            config.maximum_joint_step_rad);
      }
      if (!active_collision_scene_.enabled) {
        for (int joint = 0; joint < joint_count; ++joint) {
          q[joint] = std::clamp(
              q[joint] + delta[joint], lower[joint], upper[joint]);
        }
      } else {
        Eigen::VectorXd accepted_q = q;
        double best_step_objective = objective;
        double step_scale = 1.0;
        for (int search = 0;
             search < config.collision_line_search_steps;
             ++search) {
          Eigen::VectorXd candidate = q;
          for (int joint = 0; joint < joint_count; ++joint) {
            candidate[joint] = std::clamp(
                q[joint] + step_scale * delta[joint],
                lower[joint],
                upper[joint]);
          }
          const Evaluation candidate_evaluation =
              evaluate_recentered(
                  candidate, centroidal, anchors).second;
          double candidate_objective = 0.0;
          for (std::size_t anchor = 0;
               anchor < anchors.size();
               ++anchor) {
            const Vector3 position_error =
                anchors[anchor].target.translation -
                candidate_evaluation.anchors[anchor].translation;
            const Vector3 rotation_error =
                attitude_error(
                    anchors[anchor].target,
                    candidate_evaluation.anchors[anchor]);
            candidate_objective +=
                config.position_weight * config.position_weight *
                    position_error.squaredNorm() +
                config.attitude_weight * config.attitude_weight *
                    rotation_error.squaredNorm();
          }
          const CollisionMetrics candidate_collision =
              collision_metrics(
                  candidate_evaluation,
                  active_collision_scene_,
                  false,
                  config.collision_activation_distance_m);
          for (double clearance : candidate_collision.clearances) {
            const double penalty = std::max(
                0.0,
                config.collision_margin_m - clearance);
            candidate_objective +=
                config.collision_weight * penalty * penalty;
          }
          for (int joint = 0; joint < joint_count; ++joint) {
            const double difference =
                candidate[joint] - initial_q[joint];
            candidate_objective +=
                config.continuity_regularization_weight *
                difference * difference;
            if (pitch[joint]) {
              candidate_objective +=
                  config.pitch_joint_regularization_weight *
                  candidate[joint] * candidate[joint];
            }
          }
          if (candidate_objective < best_step_objective) {
            best_step_objective = candidate_objective;
            accepted_q = std::move(candidate);
            break;
          }
          step_scale *= 0.5;
        }
        q = std::move(accepted_q);
      }
    }
    if (best.feasible) {
      return best;
    }
    last.q = q;
    last.iterations =
        relaxed_iterations + maximum_iterations;
    return last;
  }

  int module_count_;
  int link_count_;
  int local_joint_count_;
  int root_link_;
  int base_link_;
  int base_module_;
  double module_mass_ = 0.0;
  std::vector<Joint> joints_;
  std::vector<Edge> edges_;
  std::vector<double> link_masses_;
  std::vector<Vector3> link_centers_;
  std::vector<CollisionGeometry> collision_geometry_;
  CollisionScene active_collision_scene_;
  mutable bool collision_configuration_valid_ = false;
  mutable bool collision_configuration_exact_ = false;
  mutable double collision_configuration_margin_ = std::numeric_limits<double>::quiet_NaN();
  mutable Eigen::VectorXd collision_configuration_q_;
  mutable Transform collision_configuration_pose_;
  mutable Evaluation collision_configuration_evaluation_;
  mutable CollisionFrameCache collision_configuration_frames_;
  mutable std::vector<CachedRobotPairMetric> collision_robot_metrics_;
  mutable std::unordered_map<std::uint64_t, bool>
      nominal_same_module_collision_;
  std::unordered_map<std::string, std::vector<CollisionPair>>
      collision_pair_cache_;
  bool exact_collision_geometry_loaded_ = false;
  mutable bool nominal_collision_pair_table_complete_ = false;
};

// Small dense bounded least squares for the feedback increment. QR solves only
// free columns; blocking bounds enter the active set, violated KKT bounds leave.
py::tuple bounded_least_squares(
    py::array_t<double, py::array::c_style | py::array::forcecast> a_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> b_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> lo_array,
    py::array_t<double, py::array::c_style | py::array::forcecast> hi_array,
    int maximum_iterations, double tolerance) {
  require_rank(a_array, 2, "matrix"); require_rank(b_array, 1, "rhs");
  require_rank(lo_array, 1, "lower"); require_rank(hi_array, 1, "upper");
  const int rows = a_array.shape(0), cols = a_array.shape(1);
  if (cols < 1 || rows < cols || b_array.shape(0) != rows || lo_array.shape(0) != cols || hi_array.shape(0) != cols || maximum_iterations < 1 || !(tolerance > 0) || !std::isfinite(tolerance))
    throw std::invalid_argument("invalid bounded least squares dimensions/budget");
  using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const Mat> a(a_array.data(), rows, cols);
  Eigen::Map<const Eigen::VectorXd> b(b_array.data(), rows), lo(lo_array.data(), cols), hi(hi_array.data(), cols);
  if (!a.allFinite() || !b.allFinite() || !lo.allFinite() || !hi.allFinite() || (lo.array() >= hi.array()).any())
    throw std::invalid_argument("invalid bounded least squares data/bounds");
  Eigen::VectorXd x(cols), g(cols);
  std::vector<int> active(cols, 0), free;
  bool converged = false; int iteration = 0; double optimality = std::numeric_limits<double>::infinity();
  {
    py::gil_scoped_release release;
    x = a.colPivHouseholderQr().solve(b);
    for (int i = 0; i < cols; ++i) {
      if (x[i] <= lo[i]) { x[i] = lo[i]; active[i] = -1; }
      else if (x[i] >= hi[i]) { x[i] = hi[i]; active[i] = 1; }
    }
    for (; iteration < maximum_iterations; ++iteration) {
      free.clear(); Eigen::VectorXd rhs = b;
      for (int i = 0; i < cols; ++i) {
        if (active[i] == 0) free.push_back(i); else rhs -= a.col(i) * x[i];
      }
      Eigen::VectorXd candidate = x;
      if (!free.empty()) {
        Eigen::MatrixXd af(rows, free.size());
        for (std::size_t j = 0; j < free.size(); ++j) af.col(j) = a.col(free[j]);
        Eigen::VectorXd xf = af.colPivHouseholderQr().solve(rhs);
        for (std::size_t j = 0; j < free.size(); ++j) candidate[free[j]] = xf[j];
      }
      double alpha = 1.; int blocking = -1, side = 0;
      for (int i : free) {
        if (candidate[i] < lo[i]) {
          double t = (lo[i] - x[i]) / (candidate[i] - x[i]);
          if (t < alpha) { alpha = std::max(0., t); blocking = i; side = -1; }
        } else if (candidate[i] > hi[i]) {
          double t = (hi[i] - x[i]) / (candidate[i] - x[i]);
          if (t < alpha) { alpha = std::max(0., t); blocking = i; side = 1; }
        }
      }
      x += alpha * (candidate - x);
      x = x.cwiseMax(lo).cwiseMin(hi);
      if (blocking >= 0) { x[blocking] = side < 0 ? lo[blocking] : hi[blocking]; active[blocking] = side; continue; }
      g = a.transpose() * (a * x - b);
      double violation = 0.; int release_index = -1; optimality = 0.;
      for (int i = 0; i < cols; ++i) {
        double kkt = active[i] == 0 ? std::abs(g[i]) : g[i] * active[i];
        optimality = std::max(optimality, kkt);
        if (active[i] != 0 && kkt > violation) { violation = kkt; release_index = i; }
      }
      if (violation <= tolerance || release_index < 0) { converged = optimality <= tolerance; break; }
      active[release_index] = 0;
    }
  }
  py::array_t<double> output(cols); std::copy(x.data(), x.data()+cols, output.mutable_data());
  return py::make_tuple(output, converged, std::min(iteration+1, maximum_iterations), optimality);
}

// Fixed-iteration CPU allocator core. Same six-halfspace Euclidean projection
// as batched_virtual_thrust_qp.py; no stopping or feasibility relaxation.
template <typename T>
py::array_t<T> virtual_thrust_admm(
    py::array_t<T, py::array::c_style | py::array::forcecast> lower_factor,
    py::array_t<T, py::array::c_style | py::array::forcecast> rhs_array,
    py::array_t<T, py::array::c_style | py::array::forcecast> previous_array,
    py::array_t<T, py::array::c_style | py::array::forcecast> bounds_array,
    T penalty, int iterations,
    py::array_t<T, py::array::c_style | py::array::forcecast> joint_matrix,
    py::array_t<T, py::array::c_style | py::array::forcecast> joint_lower,
    py::array_t<T, py::array::c_style | py::array::forcecast> joint_upper) {
  using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Point = Eigen::Matrix<T, 2, 1>;
  if (previous_array.ndim() != 3 || previous_array.shape(2) != 2)
    throw std::invalid_argument("invalid native ADMM previous channels");
  const int batch = previous_array.shape(0), rotors = previous_array.shape(1), width = rotors * 2;
  if (batch < 1 || rotors < 1 ||
      lower_factor.ndim() != 3 || lower_factor.shape(0) != batch ||
      lower_factor.shape(1) != width || lower_factor.shape(2) != width ||
      rhs_array.ndim() != 2 || rhs_array.shape(0) != batch || rhs_array.shape(1) != width ||
      bounds_array.ndim() != 3 || bounds_array.shape(0) != batch ||
      bounds_array.shape(1) != rotors || bounds_array.shape(2) != 5 ||
      !(penalty > T(0)) || iterations < 1) throw std::invalid_argument("invalid native ADMM inputs");
  if (joint_matrix.ndim() != 3 || joint_matrix.shape(0) != batch || joint_matrix.shape(2) != width
      || joint_lower.ndim() != 2 || joint_upper.ndim() != 2
      || joint_lower.shape(0) != batch || joint_upper.shape(0) != batch
      || joint_lower.shape(1) != joint_matrix.shape(1) || joint_upper.shape(1) != joint_matrix.shape(1))
    throw std::invalid_argument("invalid native ADMM joint constraints");
  const int joints = joint_matrix.shape(1);
  py::array_t<T> output({3, batch, width});
  T* out = output.mutable_data();
  const T eps = std::numeric_limits<T>::epsilon();
  struct Polygon {
    Point a[6]; T b[6], norm[6], tolerance;
    std::vector<Point> vertices;
    bool feasible(const Point& p) const {
      for (int i = 0; i < 6; ++i) if (a[i].dot(p) > b[i] + tolerance) return false;
      return true;
    }
    Point project(const Point& p) const {
      if (feasible(p)) return p;
      T best = std::numeric_limits<T>::infinity();
      Point result = p;
      for (int i = 0; i < 6; ++i) {
        Point v = p - ((a[i].dot(p) - b[i]) / norm[i]) * a[i];
        T d = (v - p).squaredNorm();
        if (d < best && feasible(v)) { best = d; result = v; }
      }
      for (const auto& v : vertices) {
        T d = (v - p).squaredNorm();
        if (d < best) { best = d; result = v; }
      }
      if (!std::isfinite(best)) throw std::runtime_error("empty native ADMM projection");
      return result;
    }
  };
  py::gil_scoped_release release;
  for (int env = 0; env < batch; ++env) {
    Eigen::Map<const Mat> l(lower_factor.data() + env * width * width, width, width);
    Eigen::Map<const Vec> rhs(rhs_array.data() + env * width, width);
    Eigen::Map<const Vec> previous(previous_array.data() + env * width, width);
    std::vector<Polygon> polygons(rotors);
    for (int rotor = 0; rotor < rotors; ++rotor) {
      auto& p = polygons[rotor];
      const T* v = bounds_array.data() + (env * rotors + rotor) * 5;
      // [minimum z, maximum z, maximum x, lower angle, upper angle]
      p.a[0] = Point(1,0); p.a[1] = Point(-1,0); p.a[2] = Point(0,1);
      p.a[3] = Point(0,-1); p.a[4] = Point(1,-std::tan(v[4])); p.a[5] = Point(-1,std::tan(v[3]));
      p.b[0] = p.b[1] = v[2]; p.b[2] = v[1]; p.b[3] = -v[0]; p.b[4] = p.b[5] = 0;
      T largest = 0;
      for (int i = 0; i < 6; ++i) { p.norm[i] = std::max(eps, p.a[i].squaredNorm()); largest = std::max(largest, std::abs(p.b[i])); }
      p.tolerance = T(64) * eps * (T(1) + largest);
      for (int i = 0; i < 6; ++i) for (int j = i + 1; j < 6; ++j) {
        T det = p.a[i].x() * p.a[j].y() - p.a[i].y() * p.a[j].x();
        if (std::abs(det) <= T(32) * eps) continue;
        Point vertex((p.b[i]*p.a[j].y()-p.a[i].y()*p.b[j])/det,
                     (p.a[i].x()*p.b[j]-p.b[i]*p.a[j].x())/det);
        if (p.feasible(vertex)) p.vertices.push_back(vertex);
      }
    }
    Eigen::Map<const Mat> c(joint_matrix.data() + env * joints * width, joints, width);
    Eigen::Map<const Vec> low(joint_lower.data() + env * joints, joints);
    Eigen::Map<const Vec> high(joint_upper.data() + env * joints, joints);
    Vec joint_dual = Vec::Zero(joints), y(joints), cx(joints);
    Vec z(width), x(width), dual = Vec::Zero(width), previous_z(width), solve_rhs(width);
    for (int rotor = 0; rotor < rotors; ++rotor) z.template segment<2>(rotor*2) = polygons[rotor].project(previous.template segment<2>(rotor*2));
    if (joints) y = (c * z).cwiseMax(low).cwiseMin(high);
    for (int k = 0; k < iterations; ++k) {
      solve_rhs = rhs + penalty * (z - dual);
      if (joints) solve_rhs += penalty * c.transpose() * (y - joint_dual);
      x = l.template triangularView<Eigen::Lower>().solve(solve_rhs);
      l.transpose().template triangularView<Eigen::Upper>().solveInPlace(x);
      previous_z = z;
      for (int rotor = 0; rotor < rotors; ++rotor) z.template segment<2>(rotor*2) = polygons[rotor].project(x.template segment<2>(rotor*2) + dual.template segment<2>(rotor*2));
      dual = dual + x - z;
      if (joints) {
        cx = c * x;
        y = (cx + joint_dual).cwiseMax(low).cwiseMin(high);
        joint_dual += cx - y;
      }
    }
    Eigen::Map<Vec>(out + env*width, width) = x;
    Eigen::Map<Vec>(out + batch*width + env*width, width) = z;
    Eigen::Map<Vec>(out + 2*batch*width + env*width, width) = previous_z;
  }
  return output;
}

}  // namespace

PYBIND11_MODULE(_order9_posture_native, module) {
  module.doc() = "C++/Eigen FK, collision and fixed-iteration allocation kernels";
  module.def("bounded_least_squares", &bounded_least_squares);
  module.def("virtual_thrust_admm_float32", &virtual_thrust_admm<float>);
  module.def("virtual_thrust_admm_float64", &virtual_thrust_admm<double>);
  py::class_<Kernel>(module, "Kernel")
      .def(py::init<
           int,
           int,
           int,
           int,
           int,
           int,
           py::array_t<int, py::array::c_style | py::array::forcecast>,
           py::array_t<int, py::array::c_style | py::array::forcecast>,
           py::array_t<int, py::array::c_style | py::array::forcecast>,
           py::array_t<int, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int, py::array::c_style | py::array::forcecast>,
           py::array_t<int, py::array::c_style | py::array::forcecast>,
           py::array_t<int, py::array::c_style | py::array::forcecast>,
           py::array_t<int, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>>())
      .def("evaluate", &Kernel::evaluate)
      .def("grasp_residual", &Kernel::grasp_residual)
      .def("solve_centroidal", &Kernel::solve_centroidal)
      .def(
          "configure_collision_geometry",
          &Kernel::configure_collision_geometry)
      .def(
          "configure_nominal_collision_pairs",
          &Kernel::configure_nominal_collision_pairs)
      .def(
          "nominal_collision_pairs",
          &Kernel::nominal_collision_pairs)
      .def("set_collision_scene", &Kernel::set_collision_scene)
      .def("clear_collision_scene", &Kernel::clear_collision_scene)
      .def("check_collisions", &Kernel::check_collisions);
}
