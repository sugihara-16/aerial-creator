// CPU execution of the existing articulated control-model equations.
// No physics approximations: fresh encoder poses/rates drive every call.
// Gradients and GPU execution remain in the tensor implementation.
#include <Eigen/Dense>
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
namespace py = pybind11;
template <class T> using V = Eigen::Matrix<T, 3, 1>;
template <class T> using M = Eigen::Matrix<T, 3, 3>;
template <class T>
using A = py::array_t<T, py::array::c_style | py::array::forcecast>;
template <class T> V<T> vec(const T *x) { return Eigen::Map<const V<T>>(x); }
template <class T> V<T> normal(V<T> x) {
  return x / std::max<T>(x.norm(), T(1e-12));
}
template <class T> M<T> mat(const T *x) {
  return Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor>>(x);
}
template <class T> struct Tf {
  M<T> r = M<T>::Identity();
  V<T> t = V<T>::Zero();
};
template <class T> Tf<T> compose(const Tf<T> &a, const Tf<T> &b) {
  Tf<T> o;
  o.r = a.r * b.r;
  o.t = a.t + a.r * b.t;
  return o;
}
template <class T> Tf<T> inverse(const Tf<T> &a) {
  Tf<T> o;
  o.r = a.r.transpose();
  o.t = -(o.r * a.t);
  return o;
}
template <class T> M<T> crossm(const V<T> &a) {
  M<T> o;
  o << 0, -a.z(), a.y(), a.z(), 0, -a.x(), -a.y(), a.x(), 0;
  return o;
}
template <class T> M<T> quaternion(const T *q) {
  Eigen::Matrix<T, 4, 1> v = Eigen::Map<const Eigen::Matrix<T, 4, 1>>(q);
  v /= std::max<T>(v.norm(), T(1e-12));
  T x = v[0], y = v[1], z = v[2], w = v[3];
  M<T> r;
  r << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y);
  return r;
}
struct Joint {
  int parent, child, type;
};
struct Rotor {
  int module, link, joint;
};
struct Contact {
  int module, link;
};
template <class T> class Model {
  int modules, base_module, base_link, links, joints;
  std::vector<Joint> js;
  std::vector<Rotor> rs;
  std::vector<Contact> cs;
  std::vector<int> order, load_modules, load_joints;
  std::vector<T> masses, com_weights, signed_load, contact_signed, rotor_signed,
      reactions, zsign;
  std::vector<V<T>> com_local, axis, contact_local;
  std::vector<M<T>> inertia;
  std::vector<Tf<T>> origins;
  T total;
  std::vector<T> values(py::dict d, const char *key) {
    A<T> a(d[key]);
    return {a.data(), a.data() + a.size()};
  }

public:
  Model(py::dict d) {
    modules = d["modules"].cast<int>();
    base_module = d["base_module"].cast<int>();
    base_link = d["base_link"].cast<int>();
    auto parents = d["parents"].cast<std::vector<int>>(),
         children = d["children"].cast<std::vector<int>>(),
         types = d["types"].cast<std::vector<int>>();
    order = d["order"].cast<std::vector<int>>();
    joints = parents.size();
    masses = values(d, "masses");
    links = masses.size();
    com_weights = values(d, "com_weights");
    total = d["total"].cast<T>();
    auto c = values(d, "com_local"), ins = values(d, "inertias"),
         axes = values(d, "axes"), orot = values(d, "origin_rotations"),
         ot = values(d, "origin_translations");
    if (modules < 1 || links < 1 || joints < 1 || base_module < 0 ||
        base_module >= modules || base_link < 0 || base_link >= links ||
        !std::isfinite(total) || total <= 0 ||
        children.size() != parents.size() || types.size() != parents.size() ||
        c.size() != size_t(3 * links) || ins.size() != size_t(9 * links) ||
        axes.size() != size_t(3 * joints) ||
        orot.size() != size_t(9 * joints) || ot.size() != size_t(3 * joints) ||
        com_weights.size() != size_t(joints * links))
      throw std::invalid_argument("native dynamics configuration dimensions");
    for (int j = 0; j < joints; j++)
      if (parents[j] < 0 || parents[j] >= links || children[j] < 0 ||
          children[j] >= links || types[j] < 0 || types[j] > 2)
        throw std::invalid_argument("native dynamics joint indices");
    if (order.size() != size_t(joints))
      throw std::invalid_argument("incomplete joint traversal");
    std::vector<int> visits(joints, 0);
    for (int j : order)
      if (j < 0 || j >= joints || visits[j]++)
        throw std::invalid_argument("invalid joint traversal");
    for (int l = 0; l < links; l++) {
      com_local.push_back(vec(c.data() + l * 3));
      inertia.push_back(mat(ins.data() + l * 9));
    }
    for (int j = 0; j < joints; j++) {
      js.push_back({parents[j], children[j], types[j]});
      axis.push_back(vec(axes.data() + j * 3));
      Tf<T> t;
      t.r = mat(orot.data() + 9 * j);
      t.t = vec(ot.data() + 3 * j);
      origins.push_back(t);
    }
    auto rm = d["rotor_modules"].cast<std::vector<int>>(),
         rl = d["rotor_links"].cast<std::vector<int>>(),
         rj = d["rotor_joints"].cast<std::vector<int>>();
    if (rl.size() != rm.size() || rj.size() != rm.size())
      throw std::invalid_argument("native rotor dimensions");
    for (size_t r = 0; r < rm.size(); r++)
      if (rm[r] < 0 || rm[r] >= modules || rl[r] < 0 || rl[r] >= links ||
          rj[r] < 0 || rj[r] >= joints)
        throw std::invalid_argument("native rotor index");
    for (size_t r = 0; r < rm.size(); r++)
      rs.push_back({rm[r], rl[r], rj[r]});
    reactions = values(d, "reactions");
    zsign = values(d, "zsign");
    load_modules = d["load_modules"].cast<std::vector<int>>();
    load_joints = d["load_joints"].cast<std::vector<int>>();
    signed_load = values(d, "signed_load");
    rotor_signed = values(d, "rotor_signed");
    contact_signed = values(d, "contact_signed");
    auto cm = d["contact_modules"].cast<std::vector<int>>(),
         cl = d["contact_links"].cast<std::vector<int>>();
    auto cp = values(d, "contact_positions");
    if (reactions.size() != rs.size() || zsign.size() != rs.size() ||
        load_modules.size() != load_joints.size() ||
        signed_load.size() != load_joints.size() * modules * links ||
        rotor_signed.size() != load_joints.size() * rs.size() ||
        contact_signed.size() != load_joints.size() * cm.size() ||
        cl.size() != cm.size() || cp.size() != cm.size() * 3)
      throw std::invalid_argument("native load dimensions");
    for (size_t k = 0; k < load_joints.size(); k++)
      if (load_modules[k] < 0 || load_modules[k] >= modules ||
          load_joints[k] < 0 || load_joints[k] >= joints)
        throw std::invalid_argument("native load index");
    for (size_t c = 0; c < cm.size(); c++)
      if (cm[c] < 0 || cm[c] >= modules || cl[c] < 0 || cl[c] >= links)
        throw std::invalid_argument("native contact index");
    for (size_t c = 0; c < cm.size(); c++) {
      cs.push_back({cm[c], cl[c]});
      contact_local.push_back(vec(cp.data() + 3 * c));
    }
  }
  py::dict evaluate(A<T> poses, A<T> twists, A<T> positions, A<T> rates) {
    if (poses.ndim() != 3 || poses.shape(1) != modules || poses.shape(2) != 7 ||
        twists.ndim() != 3 || twists.shape(1) != modules ||
        twists.shape(2) != 6 || positions.ndim() != 3 ||
        positions.shape(1) != modules || positions.shape(2) != joints ||
        rates.ndim() != 3 || rates.shape(1) != modules ||
        rates.shape(2) != joints || poses.shape(0) != twists.shape(0) ||
        poses.shape(0) != positions.shape(0) ||
        poses.shape(0) != rates.shape(0))
      throw std::invalid_argument("native dynamics input shape");
    int batch = poses.shape(0), rotors = rs.size(), loads = load_joints.size(),
        contacts = cs.size();
    py::dict result;
    auto output = [&](const char *key, std::vector<py::ssize_t> shape) {
      py::array_t<T> a(shape);
      std::fill(a.mutable_data(), a.mutable_data() + a.size(), T(0));
      result[key] = a;
      return a.mutable_data();
    };
    T *bp = output("body_pose_world", {batch, 7});
    T *bt = output("body_twist_world", {batch, 6});
    T *ib = output("inertia_body_matrix", {batch, 3, 3});
    T *xc = output("virtual_x_wrench_columns", {batch, rotors, 6});
    T *zc = output("virtual_z_wrench_columns", {batch, rotors, 6});
    T *angles = output("current_vectoring_angles_rad", {batch, rotors});
    T *lm = output("joint_load_matrix", {batch, loads, rotors * 2});
    T *grav = output("joint_gravity_load_nm", {batch, loads});
    T *mj = output("joint_mass_jacobian", {batch, loads, 3});
    T *am = output("joint_angular_mass_matrix", {batch, loads, 3});
    T *cent = output("joint_centrifugal_load_nm", {batch, loads});
    T *cj = output("joint_contact_jacobian", {batch, loads, contacts, 3});
    const T *p = poses.data();
    const T *v = twists.data();
    const T *q = positions.data();
    const T *qd = rates.data();
    py::gil_scoped_release release;
    for (int b = 0; b < batch; b++) {
      std::vector<Tf<T>> world(modules * links);
      std::vector<V<T>> axis_module(modules * joints), coms(modules * links);
      std::vector<M<T>> world_inertia(modules * links), mr(modules);
      V<T> center = V<T>::Zero();
      for (int m = 0; m < modules; m++) {
        mr[m] = quaternion(p + (b * modules + m) * 7 + 3);
        V<T> mt = vec(p + (b * modules + m) * 7);
        std::vector<Tf<T>> local(links);
        std::vector<V<T>> ax(joints);
        for (int j : order) {
          const auto &joint = js[j];
          Tf<T> frame = compose(local[joint.parent], origins[j]);
          ax[j] = normal<T>(frame.r * axis[j]);
          Tf<T> motion;
          T angle = q[(b * modules + m) * joints + j];
          if (joint.type == 1) {
            auto sk = crossm(normal<T>(axis[j]));
            motion.r = M<T>::Identity() + std::sin(angle) * sk +
                       (T(1) - std::cos(angle)) * (sk * sk);
          } else if (joint.type == 2)
            motion.t = angle * normal<T>(axis[j]);
          local[joint.child] = compose(frame, motion);
        }
        Tf<T> inv = inverse(local[base_link]);
        Tf<T> module;
        module.r = mr[m];
        module.t = mt;
        for (int l = 0; l < links; l++) {
          int n = m * links + l;
          world[n] = compose(module, compose(inv, local[l]));
          coms[n] = world[n].t + world[n].r * com_local[l];
          world_inertia[n] = world[n].r * inertia[l] * world[n].r.transpose();
          center += coms[n] * masses[l];
        }
        for (int j = 0; j < joints; j++)
          axis_module[m * joints + j] = normal<T>(inv.r * ax[j]);
      }
      center /= total;
      const M<T> &base_rotation = mr[base_module];
      M<T> world_to_body = base_rotation.transpose();
      std::copy(center.data(), center.data() + 3, bp + b * 7);
      std::copy(p + (b * modules + base_module) * 7 + 3,
                p + (b * modules + base_module) * 7 + 7, bp + b * 7 + 3);
      V<T> momentum = V<T>::Zero();
      M<T> body_inertia = M<T>::Zero();
      for (int m = 0; m < modules; m++) {
        V<T> linear = vec(v + (b * modules + m) * 6),
             omega = vec(v + (b * modules + m) * 6 + 3),
             translation = vec(p + (b * modules + m) * 7);
        for (int l = 0; l < links; l++) {
          int n = m * links + l;
          V<T> r = world_to_body * (coms[n] - center);
          body_inertia += world_to_body * world_inertia[n] * base_rotation +
                          masses[l] * (r.squaredNorm() * M<T>::Identity() -
                                       r * r.transpose());
          momentum += masses[l] * (linear + omega.cross(coms[n] - translation));
        }
        for (int j = 0; j < joints; j++)
          if (js[j].type) {
            V<T> a = mr[m] * axis_module[m * joints + j];
            V<T> derivative = V<T>::Zero();
            if (js[j].type == 1) {
              auto &parent = world[m * links + js[j].parent];
              V<T> origin = parent.t + parent.r * origins[j].t;
              V<T> lever = V<T>::Zero();
              for (int l = 0; l < links; l++)
                lever +=
                    (coms[m * links + l] - origin) * com_weights[j * links + l];
              derivative = a.cross(lever);
            } else {
              T sum = 0;
              for (int l = 0; l < links; l++)
                sum += com_weights[j * links + l];
              derivative = a * sum;
            }
            momentum += derivative * qd[(b * modules + m) * joints + j];
          }
      }
      V<T> velocity = momentum / total,
           omega = vec(v + (b * modules + base_module) * 6 + 3);
      std::copy(velocity.data(), velocity.data() + 3, bt + b * 6);
      std::copy(omega.data(), omega.data() + 3, bt + b * 6 + 3);
      Eigen::Map<Eigen::Matrix<T, 3, 3, Eigen::RowMajor>>(ib + b * 9) =
          body_inertia;
      std::vector<V<T>> rotor_position(rotors), rotor_x(rotors),
          rotor_z(rotors), contact_position(contacts);
      for (int r = 0; r < rotors; r++) {
        const auto &rotor = rs[r];
        int m = rotor.module, j = rotor.joint;
        rotor_position[r] = world[m * links + rotor.link].t;
        V<T> origin = world_to_body * (rotor_position[r] - center);
        V<T> zb = normal<T>(world_to_body * world[m * links + js[j].parent].r *
                            V<T>(0, 0, zsign[r]));
        V<T> ab =
            normal<T>(world_to_body * mr[m] * axis_module[m * joints + j]);
        V<T> xb = normal<T>(ab.cross(zb));
        // _wrench_column normalizes its axis a second time.
        xb = normal<T>(xb);
        zb = normal<T>(zb);
        V<T> xt = origin.cross(xb) + reactions[r] * xb,
             zt = origin.cross(zb) + reactions[r] * zb;
        std::copy(xb.data(), xb.data() + 3, xc + (b * rotors + r) * 6);
        std::copy(xt.data(), xt.data() + 3, xc + (b * rotors + r) * 6 + 3);
        std::copy(zb.data(), zb.data() + 3, zc + (b * rotors + r) * 6);
        std::copy(zt.data(), zt.data() + 3, zc + (b * rotors + r) * 6 + 3);
        rotor_x[r] = base_rotation * xb;
        rotor_z[r] = base_rotation * zb;
        angles[b * rotors + r] = q[(b * modules + m) * joints + j];
      }
      for (int c = 0; c < contacts; c++) {
        const auto &contact = cs[c];
        auto &link = world[contact.module * links + contact.link];
        contact_position[c] = link.t + link.r * contact_local[c];
      }
      for (int k = 0; k < loads; k++) {
        int m = load_modules[k], j = load_joints[k];
        auto &parent = world[m * links + js[j].parent];
        V<T> origin = parent.t + parent.r * origins[j].t;
        V<T> axis_world = mr[m] * axis_module[m * joints + j];
        V<T> mass_jac = V<T>::Zero(), angular = V<T>::Zero();
        T centrifugal = 0;
        for (int n = 0; n < modules * links; n++) {
          T sign = signed_load[k * modules * links + n];
          if (sign == 0)
            continue;
          int l = n % links;
          V<T> point = axis_world.cross(coms[n] - origin) * sign;
          V<T> lever = coms[n] - center;
          mass_jac += point * masses[l];
          angular += lever.cross(point) * masses[l] +
                     world_inertia[n] * axis_world * sign;
          V<T> force = omega.cross(omega.cross(lever)) * masses[l],
               torque = omega.cross(world_inertia[n] * omega);
          centrifugal += point.dot(force) + axis_world.dot(torque) * sign;
        }
        std::copy(mass_jac.data(), mass_jac.data() + 3,
                  mj + (b * loads + k) * 3);
        std::copy(angular.data(), angular.data() + 3, am + (b * loads + k) * 3);
        grav[b * loads + k] = T(-9.81) * mass_jac.z();
        cent[b * loads + k] = centrifugal;
        for (int r = 0; r < rotors; r++) {
          V<T> jac = axis_world.cross(rotor_position[r] - origin) +
                     axis_world * reactions[r];
          T sign = rotor_signed[k * rotors + r];
          lm[(b * loads + k) * rotors * 2 + r * 2] = jac.dot(rotor_x[r]) * sign;
          lm[(b * loads + k) * rotors * 2 + r * 2 + 1] =
              jac.dot(rotor_z[r]) * sign;
        }
        for (int c = 0; c < contacts; c++) {
          V<T> jac = axis_world.cross(contact_position[c] - origin) *
                     contact_signed[k * contacts + c];
          std::copy(jac.data(), jac.data() + 3,
                    cj + ((b * loads + k) * contacts + c) * 3);
        }
      }
    }
    return result;
  }
};
PYBIND11_MODULE(_request_dynamics_native, m) {
  py::class_<Model<float>>(m, "FloatModel")
      .def(py::init<py::dict>())
      .def("evaluate", &Model<float>::evaluate);
  py::class_<Model<double>>(m, "DoubleModel")
      .def(py::init<py::dict>())
      .def("evaluate", &Model<double>::evaluate);
}
