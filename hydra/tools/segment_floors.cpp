// Run the floor, room and building update functors on a saved scene graph.
//
//   segment_floors backend/dsg.json -c hydra_isaac_floors.yaml -o dsg_floors.json \
//       --body-height 0.59 --stairs-label 16 --floor-label 2
//
// Graphs saved before FLOORS existed keep BUILDINGS at layer 5; they are moved to the
// current layer table (FLOORS 5, CONNECTORS 4p1, BUILDINGS 6) first. The functor
// configs are read from the Hydra config under backend/update_functors/<name>.
#include <config_utilities/parsing/yaml.h>
#include <config_utilities/printing.h>
#include <glog/logging.h>
#include <hydra/backend/updates/update_buildings_functor.h>
#include <hydra/backend/updates/update_floors_functor.h>
#include <hydra/backend/updates/update_rooms_functor.h>
#include <hydra/common/shared_dsg_info.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph.h>

#include <CLI/CLI.hpp>
#include <chrono>

using namespace hydra;
using spark_dsg::DsgLayers;
using spark_dsg::SceneGraph;

int main(int argc, char* argv[]) {
  FLAGS_minloglevel = 0;
  FLAGS_logtostderr = 1;
  FLAGS_v = 1;

  CLI::App app("Segment a saved scene graph into floors and connectors");
  std::string input;
  std::string config_path;
  std::string output = "dsg_floors.json";
  double body_height = 0.0;
  std::vector<uint32_t> stairs_labels;
  std::vector<uint32_t> floor_labels;
  size_t spins = 3;
  app.add_option("graph", input)->required()->check(CLI::ExistingFile);
  app.add_option("-c,--config", config_path, "Hydra config (backend/update_functors)")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-o,--output", output);
  app.add_option("--body-height", body_height, "robot frame above the ground [m]");
  app.add_option("--stairs-label", stairs_labels, "mesh label(s) of stairs");
  app.add_option("--floor-label", floor_labels, "mesh label(s) of floor");
  app.add_option("--spins", spins, "update rounds (floors need confirm_spins)");
  CLI11_PARSE(app, argc, argv);
  google::InitGoogleLogging(argv[0]);

  auto old = SceneGraph::load(input);
  const auto old_buildings = old->getLayerKey(DsgLayers::BUILDINGS);
  if (old_buildings && old_buildings->layer == 5 && !old->getLayerKey(DsgLayers::FLOORS)) {
    LOG(INFO) << "Dropping the old BUILDINGS layer (5)";
    old->removeLayer(5);
  }

  const SharedDsgInfo::Config layers;
  auto dsg = std::make_shared<SharedDsgInfo>(layers);
  dsg->graph->mergeGraph(*old);
  if (old->hasMesh()) {
    dsg->graph->setMesh(old->mesh());
  }

  const std::string ns = "backend/update_functors/";
  auto floors_config =
      config::fromYamlFile<UpdateFloorsFunctor::Config>(config_path, ns + "floors");
  floors_config.body_height = body_height;
  floors_config.stairs_labels.insert(
      floors_config.stairs_labels.end(), stairs_labels.begin(), stairs_labels.end());
  floors_config.floor_labels.insert(
      floors_config.floor_labels.end(), floor_labels.begin(), floor_labels.end());
  const auto rooms_config =
      config::fromYamlFile<UpdateRoomsFunctor::Config>(config_path, ns + "rooms");
  LOG(INFO) << "\n" << config::toString(floors_config) << "\n" << config::toString(rooms_config);

  UpdateFloorsFunctor floors(floors_config);
  UpdateRoomsFunctor rooms(rooms_config);
  UpdateBuildingsFunctor buildings(UpdateBuildingsFunctor::Config{});

  const auto unmerged = dsg->graph->clone();
  UpdateInfo::ConstPtr info(new UpdateInfo{});
  for (size_t i = 0; i < spins; ++i) {
    const auto start = std::chrono::steady_clock::now();
    floors.call(*unmerged, *dsg, info);
    const auto mid = std::chrono::steady_clock::now();
    rooms.call(*unmerged, *dsg, info);
    buildings.call(*unmerged, *dsg, info);
    const auto end = std::chrono::steady_clock::now();
    LOG(INFO) << "spin " << i << ": floors "
              << std::chrono::duration<double, std::milli>(mid - start).count()
              << " ms, rooms + buildings "
              << std::chrono::duration<double, std::milli>(end - mid).count() << " ms";
  }

  const auto& graph = *dsg->graph;
  for (const auto& node : graph.getLayer(DsgLayers::FLOORS).nodes()) {
    LOG(INFO) << spark_dsg::NodeSymbol(node.id).str() << ":\n" << node.attributes();
  }
  for (const auto& node : graph.getLayer(DsgLayers::CONNECTORS).nodes()) {
    LOG(INFO) << spark_dsg::NodeSymbol(node.id).str() << ":\n" << node.attributes();
  }
  LOG(INFO) << graph.getLayer(DsgLayers::ROOMS).numNodes() << " room(s)";

  graph.save(output, old->hasMesh());
  LOG(INFO) << "Saved " << output;
  return 0;
}
