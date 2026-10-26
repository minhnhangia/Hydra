// Replay room detection on a saved scene graph as it grew, to see rooms appear and be
// reabsorbed.
//
//   room_history backend/dsg.json -c hydra_isaac_floors.yaml -o out_dir [--step-s 30]
//
// At each time t only the places finalised by t (last_update_time_ns <= t) are kept,
// which is what the backend had once the robot left them. For every step the global
// dilation threshold chosen by the RoomFinder and the rooms are recorded; rooms are
// tracked across steps by shared places. When two rooms that both existed for a while
// end up in one room, the event is logged with the threshold before / after and the
// bottleneck between them: the largest dilation at which they are still connected, i.e.
// the threshold they need to stay apart.
//
// Writes <out>/room_history.csv, <out>/room_events.csv and <out>/summary.yaml. The room
// finder config is read from <config> at backend/update_functors/rooms (override with
// --namespace); floor tags from UpdateFloorsFunctor are honoured with split_by_floor.
#include <config_utilities/parsing/yaml.h>
#include <config_utilities/printing.h>
#include <glog/logging.h>
#include <hydra/backend/updates/update_rooms_functor.h>
#include <hydra/rooms/graph_filtration.h>
#include <hydra/rooms/room_finder.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/scene_graph.h>

#include <CLI/CLI.hpp>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>

using namespace hydra;
using spark_dsg::DsgLayers;
using spark_dsg::NodeId;
using spark_dsg::SceneGraph;
using spark_dsg::SceneGraphLayer;
using Room = std::set<NodeId>;

namespace {

// same steps as RoomFinder::getBestComponents, but returning the chosen threshold
std::optional<FiltrationInfo> chooseThreshold(const RoomFinderConfig& config,
                                              const SceneGraphLayer& places,
                                              Filtration* filtration_out = nullptr) {
  const DistanceAdaptor get_distance;
  BarcodeTracker tracker(config.min_component_size);
  const auto filtration = getGraphFiltration(
      places,
      tracker,
      config.dilation_diff_threshold_m,
      [&](const DisjointSet& components) -> size_t {
        size_t num = 0;
        for (const auto& [id, size] : components.sizes) {
          num += size >= config.min_component_size ? 1 : 0;
        }
        return num;
      },
      false,
      get_distance);
  if (filtration_out) {
    *filtration_out = filtration;
  }

  auto window = getTrimmedFiltration(filtration,
                                     config.min_dilation_m,
                                     config.max_dilation_m,
                                     config.clip_dilation_window_to_max);
  if (window.first >= filtration.size()) {
    return std::nullopt;
  }

  if (config.clip_dilation_window_to_max) {
    const double size =
        filtration[window.second].distance - filtration[window.first].distance;
    if (size < config.min_window_size) {
      window = getTrimmedFiltration(
          filtration, config.min_dilation_m, config.max_dilation_m, false);
    }
  }

  switch (config.dilation_threshold_mode) {
    case DilationThresholdMode::REPEATED:
      return getLongestSequence(filtration, window.first, window.second);
    case DilationThresholdMode::LONGEST_LIFETIME:
      return getLongestLifetimeDilation(filtration,
                                        tracker.barcodes,
                                        config.min_lifetime_length_m,
                                        window.first,
                                        window.second);
    case DilationThresholdMode::PLATEAU_THRESHOLD:
      return getBestPlateau(
          filtration, config.min_lifetime_length_m, window.first, window.second, true);
    case DilationThresholdMode::PLATEAU:
    default:
      return getBestPlateau(
          filtration, config.plateau_ratio, window.first, window.second, false);
  }
}

// largest dilation at which a node of `a` and a node of `b` are still connected
double bottleneck(const SceneGraphLayer& places, const Room& a, const Room& b) {
  const DistanceAdaptor get_distance;
  std::map<NodeId, double> distance;
  for (const auto& node : places.nodes()) {
    if (const auto d = get_distance(node)) {
      distance[node.id] = *d;
    }
  }

  std::vector<std::tuple<double, NodeId, NodeId>> edges;
  for (const auto& edge : places.edges()) {
    const auto s = distance.find(edge.source);
    const auto t = distance.find(edge.target);
    if (s == distance.end() || t == distance.end()) {
      continue;
    }
    edges.emplace_back(std::min({get_distance(edge), s->second, t->second}),
                       edge.source,
                       edge.target);
  }
  std::sort(edges.begin(), edges.end(), [](const auto& x, const auto& y) {
    return std::get<0>(x) > std::get<0>(y);
  });

  std::map<NodeId, NodeId> parent;
  std::map<NodeId, uint8_t> mask;
  for (const auto& [id, d] : distance) {
    parent[id] = id;
    mask[id] = (a.count(id) ? 1 : 0) | (b.count(id) ? 2 : 0);
  }

  std::function<NodeId(NodeId)> find = [&](NodeId x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };

  for (const auto& [d, s, t] : edges) {
    const auto rs = find(s);
    const auto rt = find(t);
    if (rs == rt) {
      continue;
    }
    parent[rt] = rs;
    mask[rs] |= mask[rt];
    if (mask[rs] == 3) {
      return d;
    }
  }
  return 0.0;  // never connected
}

Eigen::Vector3d centroid(const SceneGraphLayer& places, const Room& room) {
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  size_t n = 0;
  for (const auto id : room) {
    if (places.hasNode(id)) {
      sum += places.getNode(id).attributes().position;
      ++n;
    }
  }
  return n ? Eigen::Vector3d(sum / n) : sum;
}

}  // namespace

int main(int argc, char* argv[]) {
  FLAGS_minloglevel = 1;
  FLAGS_logtostderr = 1;

  CLI::App app("Replay room detection over time on a saved scene graph");
  std::string input;
  std::string config_path;
  std::string ns = "backend/update_functors/rooms";
  std::string output = "room_history";
  double step_s = 30.0;
  size_t min_age = 3;
  app.add_option("graph", input)->required()->check(CLI::ExistingFile);
  app.add_option("-c,--config", config_path)->required()->check(CLI::ExistingFile);
  app.add_option("--namespace", ns, "config namespace of UpdateRoomsFunctor");
  app.add_option("-o,--output", output, "output directory");
  app.add_option("--step-s", step_s, "time between replayed room detections [s]");
  app.add_option("--min-age", min_age, "steps a room must exist to count as established");
  CLI11_PARSE(app, argc, argv);
  google::InitGoogleLogging(argv[0]);

  const auto rooms_config = config::fromYamlFile<UpdateRoomsFunctor::Config>(config_path, ns);
  const UpdateRoomsFunctor functor(rooms_config);
  const auto& config = rooms_config.room_finder;

  const auto graph = SceneGraph::load(input);
  const auto& all_places = graph->getLayer(DsgLayers::PLACES);
  const auto base = rooms_config.split_by_floor ? functor.splitByFloor(all_places)
                                                : all_places.clone();

  uint64_t t_min = std::numeric_limits<uint64_t>::max();
  uint64_t t_max = 0;
  for (const auto& node : base->nodes()) {
    const auto t = node.attributes().last_update_time_ns;
    t_min = std::min(t_min, t);
    t_max = std::max(t_max, t);
  }

  std::filesystem::create_directories(output);
  std::ofstream history(std::filesystem::path(output) / "room_history.csv");
  std::ofstream events(std::filesystem::path(output) / "room_events.csv");
  history << "t_s,places,threshold_m,components,rooms\n";
  events << "t_s,lost_places,kept_places,lost_age,threshold_before_m,threshold_after_m,"
            "bottleneck_m,x,y,z\n";

  struct Track {
    Room places;
    size_t age = 0;
  };
  std::vector<Track> tracks;
  std::optional<double> last_threshold;
  size_t num_events = 0;
  size_t steps = 0;
  double threshold_change = 0.0;
  std::vector<double> bottlenecks;

  const uint64_t step_ns = static_cast<uint64_t>(step_s * 1e9);
  for (uint64_t t = t_min + step_ns; t < t_max + step_ns; t += step_ns) {
    const auto places = base->clone([&](const spark_dsg::SceneGraphNode& node) {
      return node.attributes().last_update_time_ns <= t;
    });

    RoomFinder finder(config);
    const auto threshold = chooseThreshold(config, *places);
    finder.findRooms(*places);
    RoomFinder::ClusterMap clusters;
    finder.fillClusterMap(*places, clusters);

    std::vector<Room> rooms;
    for (const auto& [room_id, members] : clusters) {
      rooms.emplace_back(members.begin(), members.end());
    }

    const double t_s = (t - t_min) * 1e-9;
    history << t_s << "," << places->numNodes() << ","
            << (threshold ? threshold->distance : -1.0) << ","
            << (threshold ? threshold->num_components : 0) << "," << rooms.size() << "\n";
    if (threshold && last_threshold) {
      threshold_change += std::abs(threshold->distance - *last_threshold);
    }
    ++steps;

    // successor of each track: the room holding most of its places
    std::map<size_t, std::vector<size_t>> successors;
    std::vector<Track> next(rooms.size());
    for (size_t i = 0; i < rooms.size(); ++i) {
      next[i].places = rooms[i];
    }
    for (size_t k = 0; k < tracks.size(); ++k) {
      size_t best = rooms.size();
      size_t best_shared = 0;
      for (size_t i = 0; i < rooms.size(); ++i) {
        size_t shared = 0;
        for (const auto id : tracks[k].places) {
          shared += rooms[i].count(id);
        }
        if (shared > best_shared) {
          best_shared = shared;
          best = i;
        }
      }
      if (best < rooms.size() && best_shared * 2 >= tracks[k].places.size()) {
        successors[best].push_back(k);
      }
    }

    for (const auto& [room, previous] : successors) {
      // the oldest predecessor keeps the room; established others were absorbed
      size_t keeper = previous.front();
      for (const auto k : previous) {
        if (tracks[k].age > tracks[keeper].age) {
          keeper = k;
        }
      }
      next[room].age = tracks[keeper].age + 1;
      for (const auto k : previous) {
        if (k == keeper || tracks[k].age < min_age) {
          continue;
        }

        const double neck = bottleneck(*places, tracks[k].places, tracks[keeper].places);
        const auto c = centroid(*places, tracks[k].places);
        events << t_s << "," << tracks[k].places.size() << ","
               << tracks[keeper].places.size() << "," << tracks[k].age << ","
               << (last_threshold ? *last_threshold : -1.0) << ","
               << (threshold ? threshold->distance : -1.0) << "," << neck << "," << c.x()
               << "," << c.y() << "," << c.z() << "\n";
        bottlenecks.push_back(neck);
        ++num_events;
      }
    }

    tracks = next;
    if (threshold) {
      last_threshold = threshold->distance;
    }
  }

  std::ofstream summary(std::filesystem::path(output) / "summary.yaml");
  summary << "steps: " << steps << "\nfinal_rooms: " << tracks.size()
          << "\nreabsorptions: " << num_events
          << "\nfinal_threshold_m: " << (last_threshold ? *last_threshold : -1.0)
          << "\nmean_threshold_change_m: " << (steps > 1 ? threshold_change / (steps - 1) : 0.0)
          << "\nbottlenecks_m: [";
  for (size_t i = 0; i < bottlenecks.size(); ++i) {
    summary << (i ? ", " : "") << bottlenecks[i];
  }
  summary << "]\n";
  std::cout << "steps " << steps << ", final rooms " << tracks.size() << ", reabsorptions "
            << num_events << ", final threshold "
            << (last_threshold ? *last_threshold : -1.0) << " m" << std::endl;
  return 0;
}
