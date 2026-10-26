#include "hydra/backend/updates/update_floors_functor.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <tuple>

#include "hydra/common/global_info.h"
#include "hydra/utils/timing_utilities.h"

using namespace spark_dsg;

namespace hydra {
namespace {

static const auto reg = config::RegistrationWithConfig<UpdateFunctor,
                                                       UpdateFloorsFunctor,
                                                       UpdateFloorsFunctor::Config>(
    "UpdateFloorsFunctor");

using Polygon = UpdateFloorsFunctor::Polygon;
using Kind = UpdateFloorsFunctor::Sample::Kind;
using hydra::timing::ScopedTimer;

inline double deg2rad(double deg) { return deg * M_PI / 180.0; }
inline double rad2deg(double rad) { return rad * 180.0 / M_PI; }

double cross2d(const Eigen::Vector2d& o, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
}

// Andrew's monotone chain; returns the hull counter-clockwise
Polygon convexHull(std::vector<Eigen::Vector2d> points) {
  std::sort(points.begin(), points.end(), [](const auto& a, const auto& b) {
    return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
  });
  points.erase(std::unique(points.begin(), points.end()), points.end());
  if (points.size() < 3) {
    return points;
  }

  Polygon hull(2 * points.size());
  size_t k = 0;
  for (size_t i = 0; i < points.size(); ++i) {
    while (k >= 2 && cross2d(hull[k - 2], hull[k - 1], points[i]) <= 0) {
      --k;
    }
    hull[k++] = points[i];
  }

  for (size_t i = points.size() - 1, t = k + 1; i > 0; --i) {
    while (k >= t && cross2d(hull[k - 2], hull[k - 1], points[i - 1]) <= 0) {
      --k;
    }
    hull[k++] = points[i - 1];
  }

  hull.resize(k - 1);
  return hull;
}

// Hull of the points grown by a disc of the given radius (approximated by 8 points)
Polygon bufferedHull(const std::vector<Eigen::Vector2d>& points, double radius) {
  std::vector<Eigen::Vector2d> grown;
  grown.reserve(points.size() * 8);
  for (const auto& p : points) {
    for (size_t i = 0; i < 8; ++i) {
      const double theta = i * M_PI / 4.0;
      grown.push_back(p + radius * Eigen::Vector2d(std::cos(theta), std::sin(theta)));
    }
  }
  return convexHull(grown);
}

double polygonArea(const Polygon& polygon) {
  if (polygon.size() < 3) {
    return 0.0;
  }

  double area = 0.0;
  for (size_t i = 0; i < polygon.size(); ++i) {
    const auto& a = polygon[i];
    const auto& b = polygon[(i + 1) % polygon.size()];
    area += a.x() * b.y() - b.x() * a.y();
  }
  return std::abs(area) / 2.0;
}

bool contains(const Polygon& polygon, const Eigen::Vector2d& p) {
  if (polygon.size() < 3) {
    return false;
  }

  bool inside = false;
  for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
    const auto& a = polygon[i];
    const auto& b = polygon[j];
    if ((a.y() > p.y()) != (b.y() > p.y()) &&
        p.x() < (b.x() - a.x()) * (p.y() - a.y()) / (b.y() - a.y()) + a.x()) {
      inside = !inside;
    }
  }
  return inside;
}

Eigen::Vector2d centroid(const Polygon& polygon) {
  Eigen::Vector2d sum = Eigen::Vector2d::Zero();
  for (const auto& p : polygon) {
    sum += p;
  }
  return polygon.empty() ? sum : Eigen::Vector2d(sum / polygon.size());
}

std::vector<Eigen::Vector3d> toFootprint(const Polygon& polygon, double z) {
  std::vector<Eigen::Vector3d> footprint;
  for (const auto& p : polygon) {
    footprint.emplace_back(p.x(), p.y(), z);
  }
  return footprint;
}

struct Peak {
  double z;
  double weight;
};

// Peaks of a weighted 1D distribution, smoothed and non-maximum suppressed
std::vector<Peak> findPeaks(const std::vector<std::pair<double, double>>& values,
                            double resolution,
                            double sigma,
                            double min_weight,
                            double min_separation,
                            double refine_radius) {
  if (values.empty()) {
    return {};
  }

  double z_min = values.front().first;
  double z_max = z_min;
  for (const auto& [z, w] : values) {
    z_min = std::min(z_min, z);
    z_max = std::max(z_max, z);
  }

  const double pad = 3.0 * sigma + resolution;
  z_min -= pad;
  const size_t num_bins = static_cast<size_t>(std::ceil((z_max + pad - z_min) / resolution)) + 1;
  std::vector<double> bins(num_bins, 0.0);
  for (const auto& [z, w] : values) {
    bins[static_cast<size_t>((z - z_min) / resolution)] += w;
  }

  const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma / resolution)));
  std::vector<double> kernel(2 * radius + 1);
  for (int i = -radius; i <= radius; ++i) {
    const double x = i * resolution;
    kernel[i + radius] = std::exp(-0.5 * x * x / (sigma * sigma));
  }
  const double kernel_sum = std::accumulate(kernel.begin(), kernel.end(), 0.0);

  std::vector<double> smoothed(num_bins, 0.0);
  for (size_t i = 0; i < num_bins; ++i) {
    for (int k = -radius; k <= radius; ++k) {
      const int j = static_cast<int>(i) + k;
      if (j >= 0 && j < static_cast<int>(num_bins)) {
        smoothed[i] += bins[j] * kernel[k + radius] / kernel_sum;
      }
    }
  }

  // weight of a peak is the (unsmoothed) mass within the refine radius
  std::vector<Peak> candidates;
  for (size_t i = 0; i < num_bins; ++i) {
    const double left = i > 0 ? smoothed[i - 1] : 0.0;
    const double right = i + 1 < num_bins ? smoothed[i + 1] : 0.0;
    if (smoothed[i] <= 0.0 || smoothed[i] < right || smoothed[i] <= left) {
      continue;
    }

    const double z_peak = z_min + (i + 0.5) * resolution;
    double mass = 0.0;
    double weighted_z = 0.0;
    for (const auto& [z, w] : values) {
      if (std::abs(z - z_peak) <= refine_radius) {
        mass += w;
        weighted_z += w * z;
      }
    }

    if (mass >= min_weight) {
      candidates.push_back({weighted_z / mass, mass});
    }
  }

  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    return a.weight > b.weight;
  });

  std::vector<Peak> peaks;
  for (const auto& candidate : candidates) {
    const bool suppressed = std::any_of(peaks.begin(), peaks.end(), [&](const auto& p) {
      return std::abs(p.z - candidate.z) < min_separation;
    });
    if (!suppressed) {
      peaks.push_back(candidate);
    }
  }

  return peaks;
}

}  // namespace

std::optional<NodeId> getFloorTag(const NodeAttributes& attrs, const std::string& key) {
  const auto& contents = attrs.metadata.get();
  if (!contents.is_object() || !contents.contains(key) || !contents[key].is_number()) {
    return std::nullopt;
  }
  return contents[key].get<NodeId>();
}

bool isCrossFloorEdge(const EdgeAttributes& attrs) {
  const auto& contents = attrs.metadata.get();
  return contents.is_object() && contents.contains("cross_floor") &&
         contents["cross_floor"].is_boolean() && contents["cross_floor"].get<bool>();
}

void declare_config(UpdateFloorsFunctor::Config& config) {
  using namespace config;
  name("UpdateFloorsFunctor::Config");
  field(config.places_layer, "places_layer");
  field(config.assigned_layers, "assigned_layers");
  field(config.body_height, "body_height", "m");
  field(config.slope_window_m, "slope_window_m", "m");
  field(config.flat_slope_deg, "flat_slope_deg", "deg");
  field(config.flat_pitch_deg, "flat_pitch_deg", "deg");
  field(config.transition_slope_deg, "transition_slope_deg", "deg");
  field(config.transition_pitch_deg, "transition_pitch_deg", "deg");
  field(config.min_transition_rise_m, "min_transition_rise_m", "m");
  field(config.histogram_resolution_m, "histogram_resolution_m", "m");
  field(config.histogram_sigma_m, "histogram_sigma_m", "m");
  field(config.min_level_separation_m, "min_level_separation_m", "m");
  field(config.min_peak_weight, "min_peak_weight");
  field(config.place_weight, "place_weight");
  field(config.level_tolerance_m, "level_tolerance_m", "m");
  field(config.level_merge_distance_m, "level_merge_distance_m", "m");
  field(config.min_floor_area_m2, "min_floor_area_m2", "m^2");
  field(config.min_exit_path_m, "min_exit_path_m", "m");
  field(config.min_storey_height_m, "min_storey_height_m", "m");
  field(config.demote_ratio, "demote_ratio");
  field(config.confirm_spins, "confirm_spins");
  field(config.max_missed_spins, "max_missed_spins");
  field(config.assignment_tolerance_m, "assignment_tolerance_m", "m");
  field(config.connector_buffer_m, "connector_buffer_m", "m");
  field(config.connector_match_distance_m, "connector_match_distance_m", "m");
  field(config.connector_support_margin_m, "connector_support_margin_m", "m");
  field(config.stairs_labels, "stairs_labels");
  field(config.stairs_label_names, "stairs_label_names");
  field(config.floor_labels, "floor_labels");
  field(config.floor_label_names, "floor_label_names");
  field(config.mesh_every_n, "mesh_every_n");
  field(config.level_normal_z, "level_normal_z");
  field(config.walkable_normal_z, "walkable_normal_z");
  field(config.min_mesh_area_m2, "min_mesh_area_m2", "m^2");
  field(config.min_stairs_level_ratio, "min_stairs_level_ratio");
  field(config.min_tread_peaks, "min_tread_peaks");
  field(config.min_step_height_m, "min_step_height_m", "m");
  field(config.max_step_height_m, "max_step_height_m", "m");
  field(config.max_ramp_level_ratio, "max_ramp_level_ratio");
  field(config.min_ramp_slope_deg, "min_ramp_slope_deg", "deg");
  field(config.max_ramp_slope_deg, "max_ramp_slope_deg", "deg");
  field(config.min_sloped_area_m2, "min_sloped_area_m2", "m^2");
  field(config.min_stairs_surface_slope_deg, "min_stairs_surface_slope_deg", "deg");
  field(config.min_stairs_label_share, "min_stairs_label_share");
  field(config.min_stairs_slope_deg, "min_stairs_slope_deg", "deg");

  check(config.slope_window_m, GT, 0.0, "slope_window_m");
  check(config.histogram_resolution_m, GT, 0.0, "histogram_resolution_m");
  check(config.histogram_sigma_m, GT, 0.0, "histogram_sigma_m");
  check(config.level_tolerance_m, GT, 0.0, "level_tolerance_m");
  checkCondition(config.flat_slope_deg <= config.transition_slope_deg,
                 "flat_slope_deg must not exceed transition_slope_deg");
  checkCondition(config.flat_pitch_deg <= config.transition_pitch_deg,
                 "flat_pitch_deg must not exceed transition_pitch_deg");
}

UpdateFloorsFunctor::UpdateFloorsFunctor(const Config& config)
    : config(config::checkValid(config)),
      stairs_labels_(config.stairs_labels.begin(), config.stairs_labels.end()),
      floor_labels_(config.floor_labels.begin(), config.floor_labels.end()) {
  const std::set<std::string> stairs(config.stairs_label_names.begin(),
                                     config.stairs_label_names.end());
  const std::set<std::string> floors(config.floor_label_names.begin(),
                                     config.floor_label_names.end());
  for (const auto& [label, name] : GlobalInfo::instance().labelspace().label_names) {
    if (stairs.count(name)) {
      stairs_labels_.insert(label);
    }
    if (floors.count(name)) {
      floor_labels_.insert(label);
    }
  }
}

void UpdateFloorsFunctor::call(const SceneGraph&,
                               SharedDsgInfo& dsg,
                               const UpdateInfo::ConstPtr& info) const {
  auto& graph = *dsg.graph;
  if (!graph.getLayerKey(DsgLayers::FLOORS)) {
    LOG_FIRST_N(WARNING, 1) << "[floors] scene graph has no layer '" << DsgLayers::FLOORS
                            << "'; floors are disabled";
    return;
  }

  ScopedTimer timer("backend/floors", info->timestamp_ns, true, 1, false);
  ++num_spins_;
  samples_ = extractTrajectory(graph);
  classifySamples(samples_);
  const auto place_support = estimatePlaceSupport(graph);
  updateLevels(samples_, place_support, graph, info->timestamp_ns);
  updateConnectors(samples_, graph);
  writeGraph(graph, place_support);
}

std::vector<UpdateFloorsFunctor::Sample> UpdateFloorsFunctor::extractTrajectory(
    const SceneGraph& graph) const {
  std::vector<Sample> samples;
  const auto key = graph.getLayerKey(DsgLayers::AGENTS);
  if (!key) {
    return samples;
  }

  for (const auto& layer : graph.layer_partition(key->layer)) {
    for (const auto& node : layer.nodes()) {
      const auto attrs = node.tryAttributes<AgentNodeAttributes>();
      if (!attrs) {
        continue;
      }

      const Eigen::Matrix3d R = attrs->world_R_body.normalized().toRotationMatrix();
      Sample sample;
      sample.id = node.id;
      sample.stamp_ns = attrs->timestamp.count();
      // the ground is body_height below the robot frame along its z axis
      sample.ground = attrs->position - config.body_height * R.col(2);
      sample.pitch = std::asin(std::clamp(R(2, 0), -1.0, 1.0));
      samples.push_back(sample);
    }
  }

  std::sort(samples.begin(), samples.end(), [](const auto& a, const auto& b) {
    return a.stamp_ns < b.stamp_ns;
  });

  for (size_t i = 1; i < samples.size(); ++i) {
    const double step = (samples[i].ground - samples[i - 1].ground).head<2>().norm();
    samples[i].s = samples[i - 1].s + step;
  }

  for (size_t i = 0; i < samples.size(); ++i) {
    const double prev = i > 0 ? samples[i].s - samples[i - 1].s : 0.0;
    const double next = i + 1 < samples.size() ? samples[i + 1].s - samples[i].s : 0.0;
    samples[i].ds = 0.5 * (prev + next);
  }

  return samples;
}

void UpdateFloorsFunctor::classifySamples(std::vector<Sample>& samples) const {
  const double half_window = config.slope_window_m / 2.0;
  const double flat_slope = std::tan(deg2rad(config.flat_slope_deg));
  const double transition_slope = std::tan(deg2rad(config.transition_slope_deg));
  const double flat_pitch = deg2rad(config.flat_pitch_deg);
  const double transition_pitch = deg2rad(config.transition_pitch_deg);

  for (size_t i = 0; i < samples.size(); ++i) {
    size_t first = i;
    size_t last = i;
    while (first > 0 && samples[i].s - samples[first - 1].s <= half_window) {
      --first;
    }
    while (last + 1 < samples.size() && samples[last + 1].s - samples[i].s <= half_window) {
      ++last;
    }

    // least-squares fit of z over arc length
    const size_t n = last - first + 1;
    double mean_s = 0.0;
    double mean_z = 0.0;
    for (size_t j = first; j <= last; ++j) {
      mean_s += samples[j].s;
      mean_z += samples[j].ground.z();
    }
    mean_s /= n;
    mean_z /= n;

    double s_var = 0.0;
    double sz_cov = 0.0;
    for (size_t j = first; j <= last; ++j) {
      const double ds = samples[j].s - mean_s;
      s_var += ds * ds;
      sz_cov += ds * (samples[j].ground.z() - mean_z);
    }

    const double extent = samples[last].s - samples[first].s;
    auto& sample = samples[i];
    sample.slope = (n >= 3 && extent >= 0.25 * config.slope_window_m)
                       ? sz_cov / s_var
                       : std::numeric_limits<double>::quiet_NaN();

    const bool has_slope = !std::isnan(sample.slope);
    const double abs_slope = has_slope ? std::abs(sample.slope) : 0.0;
    const double abs_pitch = std::abs(sample.pitch);
    if ((has_slope && abs_slope > transition_slope) || abs_pitch > transition_pitch) {
      sample.kind = Kind::TRANSITION;
    } else if (abs_slope < flat_slope && abs_pitch < flat_pitch) {
      sample.kind = Kind::FLAT;
    } else {
      sample.kind = Kind::UNKNOWN;
    }
  }

  // runs of non-flat samples only count as a transition if they change height
  size_t i = 0;
  while (i < samples.size()) {
    if (samples[i].kind == Kind::FLAT) {
      ++i;
      continue;
    }

    size_t end = i;
    bool has_transition = false;
    while (end < samples.size() && samples[end].kind != Kind::FLAT) {
      has_transition |= samples[end].kind == Kind::TRANSITION;
      ++end;
    }

    const size_t lo = i > 0 ? i - 1 : i;
    const size_t hi = end < samples.size() ? end : end - 1;
    double z_min = samples[lo].ground.z();
    double z_max = z_min;
    for (size_t j = lo; j <= hi; ++j) {
      z_min = std::min(z_min, samples[j].ground.z());
      z_max = std::max(z_max, samples[j].ground.z());
    }

    const bool is_transition =
        has_transition && (z_max - z_min) >= config.min_transition_rise_m;
    for (size_t j = i; j < end; ++j) {
      samples[j].kind = is_transition ? Kind::TRANSITION : Kind::FLAT;
    }

    i = end;
  }
}

std::map<NodeId, double> UpdateFloorsFunctor::estimatePlaceSupport(
    const SceneGraph& graph) const {
  std::map<NodeId, double> support;
  const auto layer = graph.findLayer(config.places_layer);
  if (!layer) {
    return support;
  }

  for (const auto& node : layer->nodes()) {
    const auto attrs = node.tryAttributes<PlaceNodeAttributes>();
    if (!attrs || !attrs->real_place) {
      continue;
    }

    // highest obstacle point beneath the place's free-space sphere
    const Eigen::Vector3d& pos = attrs->position;
    const double radius = std::max(attrs->distance, 0.3) + 0.1;
    std::optional<double> best;
    for (const auto& info : attrs->voxblox_mesh_connections) {
      const Eigen::Vector3d parent(info.voxel_pos[0], info.voxel_pos[1], info.voxel_pos[2]);
      if (!parent.allFinite()) {
        continue;
      }

      if (pos.z() - parent.z() < 0.2 || (parent - pos).head<2>().norm() > radius) {
        continue;
      }

      best = best ? std::max(*best, parent.z()) : parent.z();
    }

    if (best) {
      support[node.id] = *best;
    }
  }

  return support;
}

std::vector<Polygon> UpdateFloorsFunctor::transitionFootprints(
    const std::vector<Sample>& samples) const {
  std::vector<Polygon> footprints;
  size_t i = 0;
  while (i < samples.size()) {
    if (samples[i].kind != Kind::TRANSITION) {
      ++i;
      continue;
    }

    std::vector<Eigen::Vector2d> points;
    if (i > 0) {
      points.push_back(samples[i - 1].ground.head<2>());
    }
    while (i < samples.size() && samples[i].kind == Kind::TRANSITION) {
      points.push_back(samples[i].ground.head<2>());
      ++i;
    }
    if (i < samples.size()) {
      points.push_back(samples[i].ground.head<2>());
    }

    footprints.push_back(bufferedHull(points, config.connector_buffer_m));
  }

  return footprints;
}

void UpdateFloorsFunctor::updateLevels(std::vector<Sample>& samples,
                                       const std::map<NodeId, double>& place_support,
                                       const SceneGraph& graph,
                                       uint64_t timestamp_ns) const {
  // support-height histogram: flat trajectory (by path length) and places
  std::vector<std::pair<double, double>> values;
  for (const auto& sample : samples) {
    if (sample.kind == Kind::FLAT) {
      values.emplace_back(sample.ground.z(), std::max(sample.ds, 0.02));
    }
  }
  for (const auto& [id, z] : place_support) {
    values.emplace_back(z, config.place_weight);
  }

  const auto peaks = findPeaks(values,
                               config.histogram_resolution_m,
                               config.histogram_sigma_m,
                               config.min_peak_weight,
                               config.min_level_separation_m,
                               config.level_tolerance_m / 2.0);

  // match peaks to existing levels (closest first)
  std::vector<bool> level_matched(levels_.size(), false);
  std::vector<bool> peak_matched(peaks.size(), false);
  std::vector<std::tuple<double, size_t, size_t>> pairs;
  for (size_t p = 0; p < peaks.size(); ++p) {
    for (size_t l = 0; l < levels_.size(); ++l) {
      const double dz = std::abs(peaks[p].z - levels_[l].z);
      if (dz < config.level_tolerance_m) {
        pairs.emplace_back(dz, p, l);
      }
    }
  }
  std::sort(pairs.begin(), pairs.end());
  for (const auto& [dz, p, l] : pairs) {
    if (peak_matched[p] || level_matched[l]) {
      continue;
    }
    peak_matched[p] = true;
    level_matched[l] = true;
    levels_[l].z = peaks[p].z;
    levels_[l].weight = peaks[p].weight;
    levels_[l].missed = 0;
  }

  for (size_t l = 0; l < levels_.size(); ++l) {
    if (!level_matched[l]) {
      ++levels_[l].missed;
      levels_[l].weight = 0.0;
    }
  }

  for (size_t p = 0; p < peaks.size(); ++p) {
    if (!peak_matched[p]) {
      Level level;
      level.z = peaks[p].z;
      level.weight = peaks[p].weight;
      level.first_seen_ns = timestamp_ns;
      levels_.push_back(level);
    }
  }

  // merge levels that drifted together (keep confirmed, then older)
  std::sort(levels_.begin(), levels_.end(), [](const auto& a, const auto& b) {
    if (a.confirmed != b.confirmed) {
      return a.confirmed;
    }
    return a.first_seen_ns < b.first_seen_ns;
  });
  std::vector<Level> kept;
  for (const auto& level : levels_) {
    const bool merged = std::any_of(kept.begin(), kept.end(), [&](const auto& other) {
      return std::abs(other.z - level.z) < config.level_merge_distance_m;
    });
    const bool expired = !level.confirmed && level.missed > config.max_missed_spins;
    if (!merged && !expired) {
      kept.push_back(level);
    }
  }
  levels_ = kept;
  std::sort(levels_.begin(), levels_.end(), [](const auto& a, const auto& b) {
    return a.z < b.z;
  });

  // assign flat samples to levels
  for (auto& sample : samples) {
    sample.level = -1;
    if (sample.kind != Kind::FLAT) {
      continue;
    }

    double best = config.level_tolerance_m;
    for (size_t l = 0; l < levels_.size(); ++l) {
      const double dz = std::abs(sample.ground.z() - levels_[l].z);
      if (dz < best) {
        best = dz;
        sample.level = static_cast<int>(l);
      }
    }
  }

  // floor criteria: enough area, or enough walking away from every transition
  const auto transitions = transitionFootprints(samples);
  std::vector<std::vector<Eigen::Vector2d>> points(levels_.size());
  for (auto& level : levels_) {
    level.path_m = 0.0;
    level.exit_path_m = 0.0;
    level.num_places = 0;
  }

  for (const auto& sample : samples) {
    if (sample.level < 0) {
      continue;
    }

    auto& level = levels_[sample.level];
    const Eigen::Vector2d xy = sample.ground.head<2>();
    level.path_m += sample.ds;
    points[sample.level].push_back(xy);
    const bool on_transition = std::any_of(transitions.begin(),
                                           transitions.end(),
                                           [&](const auto& fp) { return contains(fp, xy); });
    if (!on_transition) {
      level.exit_path_m += sample.ds;
    }
  }

  const auto places = graph.findLayer(config.places_layer);
  for (const auto& [id, z] : place_support) {
    for (size_t l = 0; l < levels_.size(); ++l) {
      if (std::abs(z - levels_[l].z) < config.level_tolerance_m) {
        ++levels_[l].num_places;
        if (config.place_weight > 0.0 && places && places->hasNode(id)) {
          points[l].push_back(places->getNode(id).attributes().position.head<2>());
        }
        break;
      }
    }
  }

  const auto mesh = graph.mesh();
  if (mesh && config.mesh_every_n > 0 && (num_spins_ - 1) % config.mesh_every_n == 0) {
    updateMeshFloorArea(*mesh);
  }

  for (size_t l = 0; l < levels_.size(); ++l) {
    auto& level = levels_[l];
    level.footprint = convexHull(points[l]);
    level.area_m2 = std::max(polygonArea(level.footprint), level.mesh_area_m2);
    const bool is_floor = level.exit_path_m >= config.min_exit_path_m ||
                          level.mesh_area_m2 >= config.min_floor_area_m2;
    level.streak = is_floor ? level.streak + 1 : 0;
  }

  selectFloors();
}

void UpdateFloorsFunctor::updateMeshFloorArea(const Mesh& mesh) const {
  // floor-labelled surface per level, counted in 0.25 m cells
  constexpr double cell = 0.25;
  std::vector<std::set<std::pair<int64_t, int64_t>>> cells(levels_.size());
  if (mesh.has_labels && !floor_labels_.empty()) {
    for (size_t i = 0; i < mesh.numVertices() && i < mesh.labels.size(); ++i) {
      if (!floor_labels_.count(mesh.labels[i])) {
        continue;
      }

      const Eigen::Vector3d p = mesh.pos(i).cast<double>();
      for (size_t l = 0; l < levels_.size(); ++l) {
        if (std::abs(p.z() - levels_[l].z) < 0.1) {
          cells[l].emplace(std::floor(p.x() / cell), std::floor(p.y() / cell));
          break;
        }
      }
    }
  }

  for (size_t l = 0; l < levels_.size(); ++l) {
    levels_[l].mesh_area_m2 = cells[l].size() * cell * cell;
  }
}

void UpdateFloorsFunctor::selectFloors() const {
  // Floors are at least a storey apart. Pick the set of candidate levels with the most
  // total support under that constraint (weighted interval scheduling over the levels
  // sorted by height); floors that are already confirmed get a head start, so they only
  // give way to much stronger evidence.
  const size_t n = levels_.size();
  std::vector<double> score(n, 0.0);
  for (size_t l = 0; l < n; ++l) {
    const auto& level = levels_[l];
    const bool candidate = level.confirmed || level.streak >= config.confirm_spins;
    const double support = level.path_m + level.mesh_area_m2;
    score[l] = candidate ? (level.confirmed ? config.demote_ratio : 1.0) * support : 0.0;
  }

  // best[i]: best total over levels [0, i); prev[l]: levels below l that stay compatible
  std::vector<double> best(n + 1, 0.0);
  std::vector<bool> take(n, false);
  std::vector<size_t> prev(n, 0);
  for (size_t l = 0; l < n; ++l) {
    size_t j = l;
    while (j > 0 && levels_[l].z - levels_[j - 1].z < config.min_storey_height_m) {
      --j;
    }
    prev[l] = j;
    const double with = score[l] > 0.0 ? score[l] + best[j] : 0.0;
    take[l] = with > best[l];
    best[l + 1] = take[l] ? with : best[l];
  }

  std::vector<bool> selected(n, false);
  for (size_t i = n; i > 0;) {
    if (take[i - 1]) {
      selected[i - 1] = true;
      i = prev[i - 1];
    } else {
      --i;
    }
  }

  for (size_t l = 0; l < n; ++l) {
    auto& level = levels_[l];
    if (level.confirmed && !selected[l]) {
      VLOG(1) << "[floors] " << NodeSymbol(*level.node_id).str() << " at z=" << level.z
              << " is now a landing";
    }
    level.confirmed = selected[l];
  }

  // new floors are numbered bottom-up (levels_ is sorted by height)
  for (auto& level : levels_) {
    if (level.confirmed && !level.node_id) {
      level.node_id = NodeSymbol('F', next_floor_index_++);
      VLOG(1) << "[floors] floor " << NodeSymbol(*level.node_id).str() << " at z="
              << level.z << " (mesh " << level.mesh_area_m2 << " m^2, path " << level.path_m
              << " m, exit path " << level.exit_path_m << " m)";
    }
  }
}

void UpdateFloorsFunctor::updateConnectors(const std::vector<Sample>& samples,
                                           const SceneGraph& graph) const {
  // visits: runs of flat samples on the same level
  struct Visit {
    int level;
    size_t start;
    size_t end;
  };

  std::vector<Visit> visits;
  for (size_t i = 0; i < samples.size(); ++i) {
    if (samples[i].level < 0) {
      continue;
    }

    if (!visits.empty() && visits.back().level == samples[i].level &&
        visits.back().end + 1 == i) {
      visits.back().end = i;
    } else {
      visits.push_back({samples[i].level, i, i});
    }
  }

  // a chain goes from one confirmed floor to another, possibly via landings
  struct Chain {
    int lower;
    int upper;
    std::vector<size_t> samples;
    Eigen::Vector2d lower_point;
    Eigen::Vector2d upper_point;
    double vertical = 0.0;
    double horizontal = 0.0;
    std::set<int> landings;
  };

  std::vector<Chain> chains;
  std::optional<size_t> last_floor;
  for (size_t v = 0; v < visits.size(); ++v) {
    if (!levels_[visits[v].level].confirmed) {
      continue;
    }

    if (last_floor && visits[*last_floor].level != visits[v].level) {
      const auto& from = visits[*last_floor];
      const auto& to = visits[v];
      Chain chain;
      const bool up = levels_[to.level].z > levels_[from.level].z;
      chain.lower = up ? from.level : to.level;
      chain.upper = up ? to.level : from.level;
      const auto& from_point = samples[from.end].ground.head<2>();
      const auto& to_point = samples[to.start].ground.head<2>();
      chain.lower_point = up ? from_point : to_point;
      chain.upper_point = up ? to_point : from_point;

      bool has_transition = false;
      for (size_t i = from.end; i <= to.start; ++i) {
        if (i > from.end && i < to.start) {
          chain.samples.push_back(i);
        }

        const auto& sample = samples[i];
        has_transition |= sample.kind == Kind::TRANSITION;
        if (sample.level >= 0 && !levels_[sample.level].confirmed) {
          chain.landings.insert(sample.level);
        }

        if (i > from.end &&
            (sample.kind == Kind::TRANSITION || samples[i - 1].kind == Kind::TRANSITION)) {
          chain.vertical += std::abs(sample.ground.z() - samples[i - 1].ground.z());
          chain.horizontal += sample.s - samples[i - 1].s;
        }
      }

      if (has_transition) {
        chains.push_back(chain);
      }
    }

    last_floor = v;
  }

  // traversals of the same floor pair in the same place form one connector
  struct Candidate {
    int lower;
    int upper;
    std::vector<const Chain*> chains;
    std::vector<Eigen::Vector2d> points;
  };

  std::vector<Candidate> candidates;
  for (const auto& chain : chains) {
    std::vector<Eigen::Vector2d> points{chain.lower_point, chain.upper_point};
    for (const auto i : chain.samples) {
      points.push_back(samples[i].ground.head<2>());
    }

    const Eigen::Vector2d center = centroid(points);
    Candidate* match = nullptr;
    for (auto& candidate : candidates) {
      if (candidate.lower == chain.lower && candidate.upper == chain.upper &&
          (centroid(candidate.points) - center).norm() < config.connector_match_distance_m) {
        match = &candidate;
        break;
      }
    }

    if (!match) {
      candidates.push_back({chain.lower, chain.upper, {}, {}});
      match = &candidates.back();
    }

    match->chains.push_back(&chain);
    match->points.insert(match->points.end(), points.begin(), points.end());
  }

  // keep connector ids by matching to the previous spin
  std::vector<Connector> previous = std::move(connectors_);
  connectors_.clear();
  std::vector<bool> used(previous.size(), false);
  const auto places = graph.findLayer(config.places_layer);
  for (const auto& candidate : candidates) {
    Connector connector;
    connector.lower = candidate.lower;
    connector.upper = candidate.upper;
    connector.footprint = bufferedHull(candidate.points, config.connector_buffer_m);
    const Eigen::Vector2d center = centroid(connector.footprint);
    const auto lower_id = *levels_[candidate.lower].node_id;
    const auto upper_id = *levels_[candidate.upper].node_id;

    for (size_t p = 0; p < previous.size(); ++p) {
      const auto& prev = previous[p];
      if (used[p] || prev.attrs.lower_floor != lower_id ||
          prev.attrs.upper_floor != upper_id ||
          (centroid(prev.footprint) - center).norm() >= config.connector_match_distance_m) {
        continue;
      }
      used[p] = true;
      connector.node_id = prev.node_id;
      connector.mesh = prev.mesh;
      break;
    }

    if (!connector.node_id) {
      connector.node_id = NodeSymbol('X', next_connector_index_++);
    }

    auto& attrs = connector.attrs;
    attrs.lower_floor = lower_id;
    attrs.upper_floor = upper_id;
    attrs.z_bottom = levels_[candidate.lower].z;
    attrs.z_top = levels_[candidate.upper].z;
    attrs.rise = attrs.z_top - attrs.z_bottom;

    Eigen::Vector2d entry = Eigen::Vector2d::Zero();
    Eigen::Vector2d exit = Eigen::Vector2d::Zero();
    double vertical = 0.0;
    double horizontal = 0.0;
    std::set<int> landings;
    for (const auto* chain : candidate.chains) {
      entry += chain->lower_point;
      exit += chain->upper_point;
      vertical += chain->vertical;
      horizontal += chain->horizontal;
      landings.insert(chain->landings.begin(), chain->landings.end());
      connector.samples.insert(
          connector.samples.end(), chain->samples.begin(), chain->samples.end());
    }

    const double num_chains = candidate.chains.size();
    attrs.entry << entry / num_chains, attrs.z_bottom;
    attrs.exit << exit / num_chains, attrs.z_top;
    attrs.run = horizontal / num_chains;
    attrs.slope_deg = horizontal > 0.0 ? rad2deg(std::atan2(vertical, horizontal)) : 90.0;
    attrs.landing_z.clear();
    for (const auto l : landings) {
      attrs.landing_z.push_back(levels_[l].z);
    }

    attrs.footprint = toFootprint(connector.footprint, attrs.z_bottom);
    attrs.position << center, 0.5 * (attrs.z_bottom + attrs.z_top);
    attrs.name = NodeSymbol(connector.node_id).str();
    attrs.traversed = true;
    attrs.source_flags = ConnectorNodeAttributes::TRAJECTORY;

    // usable width from the clearance of the free-space places inside
    std::vector<double> widths;
    if (places) {
      for (const auto& node : places->nodes()) {
        const auto place = node.tryAttributes<PlaceNodeAttributes>();
        const auto& pos = node.attributes().position;
        if (!place || pos.z() <= attrs.z_bottom || pos.z() >= attrs.z_top + 1.0 ||
            !contains(connector.footprint, pos.head<2>())) {
          continue;
        }
        widths.push_back(2.0 * place->distance);
      }
    }
    if (!widths.empty()) {
      std::nth_element(widths.begin(), widths.begin() + widths.size() / 2, widths.end());
      attrs.width = widths[widths.size() / 2];
    }

    const auto mesh = graph.mesh();
    const bool mesh_due = config.mesh_every_n > 0 &&
                          (!connector.mesh || (num_spins_ - 1) % config.mesh_every_n == 0);
    if (mesh && mesh_due) {
      connector.mesh = computeMeshEvidence(*mesh, connector);
    }

    classifyConnector(connector);
    connectors_.push_back(connector);
  }
}

UpdateFloorsFunctor::MeshEvidence UpdateFloorsFunctor::computeMeshEvidence(
    const Mesh& mesh, const Connector& connector) const {
  MeshEvidence evidence;
  const auto& attrs = connector.attrs;
  Eigen::Vector2d min = Eigen::Vector2d::Constant(std::numeric_limits<double>::max());
  Eigen::Vector2d max = -min;
  for (const auto& p : connector.footprint) {
    min = min.cwiseMin(p);
    max = max.cwiseMax(p);
  }

  // surfaces strictly between the two floors
  const double z_low = attrs.z_bottom + 0.05;
  const double z_high = attrs.z_top - 0.05;
  const auto& stairs = stairs_labels_;
  double level_area = 0.0;
  double stairs_area = 0.0;
  double sloped_angle = 0.0;
  std::vector<std::pair<double, double>> treads;
  for (size_t f = 0; f < mesh.numFaces(); ++f) {
    const auto& face = mesh.face(f);
    if (face[0] >= mesh.numVertices() || face[1] >= mesh.numVertices() ||
        face[2] >= mesh.numVertices()) {
      continue;
    }

    const Eigen::Vector3d a = mesh.pos(face[0]).cast<double>();
    const Eigen::Vector3d b = mesh.pos(face[1]).cast<double>();
    const Eigen::Vector3d c = mesh.pos(face[2]).cast<double>();
    const Eigen::Vector3d center = (a + b + c) / 3.0;
    if (center.z() <= z_low || center.z() >= z_high || center.x() < min.x() ||
        center.y() < min.y() || center.x() > max.x() || center.y() > max.y() ||
        !contains(connector.footprint, center.head<2>())) {
      continue;
    }

    const Eigen::Vector3d normal = (b - a).cross(c - a);
    const double norm = normal.norm();
    if (norm < 1.0e-9) {
      continue;
    }

    const double area = 0.5 * norm;
    const double nz = std::abs(normal.z()) / norm;
    if (nz < config.walkable_normal_z) {
      continue;
    }

    evidence.walkable_area += area;
    if (mesh.has_labels && face[0] < mesh.labels.size() && stairs.count(mesh.labels[face[0]])) {
      stairs_area += area;
    }

    if (nz >= config.level_normal_z) {
      level_area += area;
      treads.emplace_back(center.z(), area);
    } else {
      evidence.sloped_area += area;
      sloped_angle += area * std::acos(nz);
    }
  }

  if (evidence.sloped_area > 0.0) {
    evidence.sloped_slope_deg = rad2deg(sloped_angle / evidence.sloped_area);
  }

  if (evidence.walkable_area <= 0.0) {
    return evidence;
  }

  evidence.level_ratio = level_area / evidence.walkable_area;
  evidence.stairs_share = stairs_area / evidence.walkable_area;

  // treads are level surfaces at evenly spaced heights
  const auto peaks = findPeaks(treads,
                               0.02,
                               0.02,
                               0.03 * level_area,
                               0.5 * config.min_step_height_m,
                               0.03);
  std::vector<double> heights;
  for (const auto& peak : peaks) {
    heights.push_back(peak.z);
  }
  std::sort(heights.begin(), heights.end());
  evidence.num_treads = heights.size();
  if (heights.size() >= 2) {
    std::vector<double> steps;
    for (size_t i = 1; i < heights.size(); ++i) {
      steps.push_back(heights[i] - heights[i - 1]);
    }
    std::nth_element(steps.begin(), steps.begin() + steps.size() / 2, steps.end());
    evidence.step_height = steps[steps.size() / 2];
  }

  return evidence;
}

void UpdateFloorsFunctor::classifyConnector(Connector& connector) const {
  using Type = ConnectorNodeAttributes::Type;
  auto& attrs = connector.attrs;
  attrs.connector_type = Type::UNKNOWN;
  attrs.confidence = 0.3;
  attrs.step_height = 0.0;
  attrs.num_steps = 0;

  const auto& mesh = connector.mesh;
  if (mesh && mesh->walkable_area >= config.min_mesh_area_m2) {
    VLOG(1) << "[floors] " << NodeSymbol(connector.node_id).str() << " mesh: walkable "
            << mesh->walkable_area << " m^2, level ratio " << mesh->level_ratio << ", "
            << mesh->num_treads << " tread(s), step " << mesh->step_height
            << " m, stairs share " << mesh->stairs_share << ", sloped "
            << mesh->sloped_area << " m^2 at " << mesh->sloped_slope_deg << " deg";
    const bool steps_ok = mesh->step_height >= config.min_step_height_m &&
                          mesh->step_height <= config.max_step_height_m;
    const bool sloped = mesh->sloped_area >= config.min_sloped_area_m2;
    if (mesh->level_ratio >= config.min_stairs_level_ratio &&
        mesh->num_treads >= config.min_tread_peaks && steps_ok) {
      // resolved treads at regular heights
      attrs.connector_type = Type::STAIRS;
      attrs.confidence = 0.9;
      attrs.source_flags |= ConnectorNodeAttributes::GEOMETRY;
      attrs.step_height = mesh->step_height;
    } else if (mesh->stairs_share >= config.min_stairs_label_share) {
      attrs.connector_type = Type::STAIRS;
      attrs.confidence = 0.8;
      attrs.source_flags |= ConnectorNodeAttributes::SEMANTIC;
    } else if (sloped && mesh->sloped_slope_deg >= config.min_stairs_surface_slope_deg) {
      // coarse voxels smooth steps into a slope at the stair pitch
      attrs.connector_type = Type::STAIRS;
      attrs.confidence = 0.75;
      attrs.source_flags |= ConnectorNodeAttributes::GEOMETRY;
    } else if (sloped && mesh->sloped_slope_deg >= config.min_ramp_slope_deg &&
               mesh->sloped_slope_deg <= config.max_ramp_slope_deg &&
               mesh->level_ratio <= config.max_ramp_level_ratio) {
      attrs.connector_type = Type::RAMP;
      attrs.confidence = 0.8;
      attrs.source_flags |= ConnectorNodeAttributes::GEOMETRY;
    }
  }

  if (attrs.connector_type == Type::UNKNOWN) {
    if (attrs.slope_deg >= config.min_stairs_slope_deg) {
      attrs.connector_type = Type::STAIRS;
      attrs.confidence = 0.6;
    } else if (attrs.slope_deg >= config.min_ramp_slope_deg &&
               attrs.slope_deg <= config.max_ramp_slope_deg) {
      attrs.connector_type = Type::RAMP;
      attrs.confidence = 0.6;
    }
  }

  if (attrs.connector_type == Type::STAIRS && attrs.step_height > 0.0) {
    attrs.num_steps = static_cast<uint64_t>(std::round(attrs.rise / attrs.step_height));
  }
}

void UpdateFloorsFunctor::writeGraph(SceneGraph& graph,
                                     const std::map<NodeId, double>& place_support) const {
  // floors, ranked by height
  std::set<NodeId> floor_ids;
  int64_t index = 0;
  for (const auto& level : levels_) {
    if (!level.confirmed) {
      continue;
    }

    const auto id = *level.node_id;
    floor_ids.insert(id);
    if (!graph.hasNode(id)) {
      graph.emplaceNode(DsgLayers::FLOORS, id, std::make_unique<FloorNodeAttributes>());
    }

    auto& attrs = graph.getNode(id).attributes<FloorNodeAttributes>();
    attrs.index = index++;
    attrs.floor_z = level.z;
    attrs.footprint = toFootprint(level.footprint, level.z);
    attrs.area_m2 = level.area_m2;
    attrs.trajectory_support_m = level.path_m;
    attrs.place_support = level.num_places;
    attrs.confirmed = true;
    attrs.first_seen_ns = level.first_seen_ns;
    attrs.position << centroid(level.footprint), level.z;
    attrs.name = "floor_" + std::to_string(attrs.index);
    attrs.is_active = false;
  }

  std::vector<NodeId> stale;
  for (const auto& node : graph.getLayer(DsgLayers::FLOORS).nodes()) {
    if (!floor_ids.count(node.id)) {
      stale.push_back(node.id);
    }
  }

  // connectors, attached to both of their floors
  const bool has_connectors = graph.getLayerKey(DsgLayers::CONNECTORS).has_value();
  std::set<NodeId> connector_ids;
  if (has_connectors) {
    for (const auto& connector : connectors_) {
      const auto id = connector.node_id;
      connector_ids.insert(id);
      if (!graph.hasNode(id)) {
        graph.emplaceNode(
            DsgLayers::CONNECTORS, id, std::make_unique<ConnectorNodeAttributes>());
      }

      auto& attrs = graph.getNode(id).attributes<ConnectorNodeAttributes>();
      attrs = connector.attrs;
      attrs.is_active = false;
      graph.insertEdge(connector.attrs.lower_floor, id);
      graph.insertEdge(connector.attrs.upper_floor, id);
    }

    for (const auto& node : graph.getLayer(DsgLayers::CONNECTORS).nodes()) {
      if (!connector_ids.count(node.id)) {
        stale.push_back(node.id);
      }
    }
  }

  for (const auto id : stale) {
    graph.removeNode(id);
  }

  // assign nodes to a floor or a connector
  std::map<NodeId, NodeId> node_floor;
  std::map<NodeId, NodeId> node_connector;
  for (const auto& [layer_name, offset] : config.assigned_layers) {
    const auto layer = graph.findLayer(layer_name);
    if (!layer) {
      continue;
    }

    const bool is_places = layer_name == config.places_layer;
    for (const auto& node : layer->nodes()) {
      auto& attrs = node.attributes();
      const auto& pos = attrs.position;
      const auto support_iter = is_places ? place_support.find(node.id) : place_support.end();
      const bool has_support = support_iter != place_support.end();
      // without a measured surface, a place center is above its floor and its
      // floor is at most (center - clearance)
      const double floor_z = has_support ? support_iter->second : pos.z() - offset;
      double connector_z = floor_z;
      if (!has_support && is_places) {
        const auto place = node.tryAttributes<PlaceNodeAttributes>();
        connector_z = place ? pos.z() - place->distance : pos.z();
      }

      std::optional<NodeId> floor;
      for (const auto& level : levels_) {
        if (level.confirmed && level.z - config.assignment_tolerance_m <= floor_z) {
          floor = level.node_id;
        }
      }

      std::optional<NodeId> connector;
      for (const auto& candidate : connectors_) {
        const auto& c = candidate.attrs;
        if (connector_z > c.z_bottom + config.connector_support_margin_m &&
            connector_z < c.z_top - config.connector_support_margin_m &&
            contains(candidate.footprint, pos.head<2>())) {
          connector = candidate.node_id;
          break;
        }
      }

      // metadata is JSON: only touch it when a tag changes
      const auto connector_tag = has_connectors ? connector : std::nullopt;
      if (getFloorTag(attrs, "floor_id") != floor ||
          getFloorTag(attrs, "connector_id") != connector_tag) {
        nlohmann::json tags;
        tags["floor_id"] = floor ? nlohmann::json(*floor) : nlohmann::json(nullptr);
        tags["connector_id"] =
            connector_tag ? nlohmann::json(*connector_tag) : nlohmann::json(nullptr);
        attrs.metadata.add(tags);
      }
      if (floor) {
        node_floor[node.id] = *floor;
      }
      if (connector && has_connectors) {
        node_connector[node.id] = *connector;
      }
    }
  }

  // connectors are the parents of the nodes inside them
  if (has_connectors) {
    for (const auto id : connector_ids) {
      std::vector<NodeId> to_remove;
      for (const auto child : graph.getNode(id).children()) {
        const auto iter = node_connector.find(child);
        if (iter == node_connector.end() || iter->second != id) {
          to_remove.push_back(child);
        }
      }
      for (const auto child : to_remove) {
        graph.removeEdge(id, child);
      }
    }

    for (const auto& [node_id, connector_id] : node_connector) {
      if (!graph.hasEdge(connector_id, node_id)) {
        graph.insertEdge(connector_id, node_id, nullptr, true);
      }
    }
  }

  // edges between two floors that do not go through a connector
  size_t num_cross = 0;
  for (const auto& [layer_name, offset] : config.assigned_layers) {
    const auto layer = graph.findLayer(layer_name);
    if (!layer) {
      continue;
    }

    for (const auto& edge : layer->edges()) {
      const auto s_floor = node_floor.find(edge.source);
      const auto t_floor = node_floor.find(edge.target);
      const bool in_connector =
          node_connector.count(edge.source) || node_connector.count(edge.target);
      const bool cross = s_floor != node_floor.end() && t_floor != node_floor.end() &&
                         s_floor->second != t_floor->second && !in_connector;
      // a missing tag means false, so most edges are never written
      if (cross != isCrossFloorEdge(*edge.info)) {
        edge.info->metadata.add(nlohmann::json{{"cross_floor", cross}});
      }
      num_cross += cross ? 1 : 0;
    }
  }

  VLOG(2) << "[floors] " << floor_ids.size() << " floor(s), " << connector_ids.size()
          << " connector(s), " << num_cross << " cross-floor edge(s)";
}

}  // namespace hydra
