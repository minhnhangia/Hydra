#include <gtest/gtest.h>
#include <hydra/backend/updates/update_floors_functor.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph.h>

#include <cmath>
#include <nlohmann/json.hpp>

#include "hydra_test/shared_dsg_fixture.h"

using namespace spark_dsg;

namespace hydra {
namespace {

using Type = ConnectorNodeAttributes::Type;

// walks a polyline, dropping an agent pose every `spacing` meters
struct TrajectoryBuilder {
  explicit TrajectoryBuilder(SceneGraph& graph) : graph(graph) {}

  void walk(const std::vector<Eigen::Vector3d>& waypoints, double spacing = 0.4) {
    for (size_t i = 0; i + 1 < waypoints.size(); ++i) {
      const Eigen::Vector3d delta = waypoints[i + 1] - waypoints[i];
      const size_t steps = std::max<size_t>(1, std::ceil(delta.head<2>().norm() / spacing));
      for (size_t k = 0; k < steps; ++k) {
        add(waypoints[i] + delta * (static_cast<double>(k) / steps));
      }
    }
    add(waypoints.back());
  }

  void add(const Eigen::Vector3d& position) {
    auto attrs = std::make_unique<AgentNodeAttributes>(std::chrono::nanoseconds(stamp_ns),
                                                       Eigen::Quaterniond::Identity(),
                                                       position,
                                                       0);
    graph.emplaceNode(2, NodeSymbol('a', index++), std::move(attrs), 'a');
    stamp_ns += 1000000000;
  }

  SceneGraph& graph;
  size_t index = 0;
  uint64_t stamp_ns = 0;
};

void addPlace(SceneGraph& graph,
              NodeId id,
              const Eigen::Vector3d& pos,
              double distance,
              std::optional<double> floor_z = std::nullopt) {
  auto attrs = std::make_unique<PlaceNodeAttributes>(distance, 0);
  attrs->position = pos;
  if (floor_z) {
    auto& info = attrs->voxblox_mesh_connections.emplace_back();
    info.voxel_pos[0] = pos.x();
    info.voxel_pos[1] = pos.y();
    info.voxel_pos[2] = *floor_z;
  }
  graph.emplaceNode(DsgLayers::PLACES, id, std::move(attrs));
}

// two floors joined by a straight flight up to a landing and a second flight
void addStairsScene(SceneGraph& graph) {
  TrajectoryBuilder traj(graph);
  traj.walk({{0.0, 0.0, 0.0},
             {8.0, 0.0, 0.0},
             {11.0, 0.0, 1.65},
             {12.5, 0.0, 1.65},
             {15.5, 0.0, 3.3},
             {25.0, 0.0, 3.3}});
}

std::optional<NodeId> metadataId(const SceneGraph& graph, NodeId node, const char* key) {
  const auto& contents = graph.getNode(node).attributes().metadata.get();
  if (!contents.contains(key) || contents[key].is_null()) {
    return std::nullopt;
  }
  return contents[key].get<NodeId>();
}

UpdateFloorsFunctor::Config testConfig() {
  UpdateFloorsFunctor::Config config;
  config.mesh_every_n = 1;
  return config;
}

void spin(const UpdateFloorsFunctor& functor, SharedDsgInfo& dsg, size_t times = 3) {
  const auto unmerged = dsg.graph->clone();
  UpdateInfo::ConstPtr info(new UpdateInfo{});
  for (size_t i = 0; i < times; ++i) {
    functor.call(*unmerged, dsg, info);
  }
}

}  // namespace

TEST(UpdateFloorsFunctor, SingleFloor) {
  auto dsg = test::makeSharedDsg();
  TrajectoryBuilder traj(*dsg->graph);
  traj.walk({{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}, {10.0, 10.0, 0.0}, {0.0, 10.0, 0.0}});

  UpdateFloorsFunctor functor(testConfig());
  spin(functor, *dsg);

  const auto& floors = dsg->graph->getLayer(DsgLayers::FLOORS);
  ASSERT_EQ(floors.numNodes(), 1u);
  const auto& attrs = dsg->graph->getNode("F0"_id).attributes<FloorNodeAttributes>();
  EXPECT_NEAR(attrs.floor_z, 0.0, 0.05);
  EXPECT_EQ(attrs.index, 0);
  EXPECT_NEAR(attrs.trajectory_support_m, 30.0, 0.5);
  EXPECT_EQ(dsg->graph->getLayer(DsgLayers::CONNECTORS).numNodes(), 0u);
}

TEST(UpdateFloorsFunctor, FloorsNeedConfirmation) {
  auto dsg = test::makeSharedDsg();
  TrajectoryBuilder traj(*dsg->graph);
  traj.walk({{0.0, 0.0, 0.0}, {10.0, 0.0, 0.0}});

  UpdateFloorsFunctor functor(testConfig());
  spin(functor, *dsg, 2);
  EXPECT_EQ(dsg->graph->getLayer(DsgLayers::FLOORS).numNodes(), 0u);
  spin(functor, *dsg, 1);
  EXPECT_EQ(dsg->graph->getLayer(DsgLayers::FLOORS).numNodes(), 1u);
}

TEST(UpdateFloorsFunctor, StairsWithLanding) {
  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;
  addStairsScene(graph);

  // places on each floor, one in the stairwell and one edge through the void
  addPlace(graph, "p0"_id, {4.0, 0.0, 1.0}, 0.9, 0.0);
  addPlace(graph, "p1"_id, {20.0, 0.0, 4.3}, 0.9, 3.3);
  addPlace(graph, "p2"_id, {10.0, 0.0, 2.2}, 0.6, 1.2);
  addPlace(graph, "p3"_id, {7.0, 0.0, 2.5}, 1.5, 0.0);
  addPlace(graph, "p4"_id, {7.5, 0.5, 4.0}, 0.6, 3.3);
  graph.insertEdge("p3"_id, "p4"_id);  // lower floor to upper floor through the air
  graph.insertEdge("p0"_id, "p3"_id);  // same floor

  UpdateFloorsFunctor functor(testConfig());
  spin(functor, *dsg);

  const auto& floors = graph.getLayer(DsgLayers::FLOORS);
  ASSERT_EQ(floors.numNodes(), 2u);
  const auto& lower = graph.getNode("F0"_id).attributes<FloorNodeAttributes>();
  const auto& upper = graph.getNode("F1"_id).attributes<FloorNodeAttributes>();
  EXPECT_NEAR(lower.floor_z, 0.0, 0.05);
  EXPECT_NEAR(upper.floor_z, 3.3, 0.05);
  EXPECT_EQ(lower.index, 0);
  EXPECT_EQ(upper.index, 1);

  const auto& connectors = graph.getLayer(DsgLayers::CONNECTORS);
  ASSERT_EQ(connectors.numNodes(), 1u);
  const auto& node = graph.getNode("X0"_id);
  const auto& connector = node.attributes<ConnectorNodeAttributes>();
  EXPECT_EQ(connector.connector_type, Type::STAIRS);
  EXPECT_EQ(connector.lower_floor, "F0"_id);
  EXPECT_EQ(connector.upper_floor, "F1"_id);
  EXPECT_NEAR(connector.rise, 3.3, 0.1);
  ASSERT_EQ(connector.landing_z.size(), 1u);
  EXPECT_NEAR(connector.landing_z[0], 1.65, 0.05);
  EXPECT_GT(connector.slope_deg, 20.0);
  EXPECT_TRUE(connector.traversed);
  EXPECT_EQ(node.parents(), (std::set<NodeId>{"F0"_id, "F1"_id}));

  EXPECT_EQ(metadataId(graph, "p0"_id, "floor_id"), "F0"_id);
  EXPECT_EQ(metadataId(graph, "p1"_id, "floor_id"), "F1"_id);
  EXPECT_EQ(metadataId(graph, "p2"_id, "connector_id"), "X0"_id);
  EXPECT_FALSE(metadataId(graph, "p0"_id, "connector_id"));
  EXPECT_TRUE(graph.hasEdge("X0"_id, "p2"_id));

  const auto& void_edge = graph.getEdge("p3"_id, "p4"_id);
  EXPECT_TRUE(isCrossFloorEdge(*void_edge.info));
  const auto& floor_edge = graph.getEdge("p0"_id, "p3"_id);
  EXPECT_FALSE(isCrossFloorEdge(*floor_edge.info));
}

TEST(UpdateFloorsFunctor, LandingsAreNotFloors) {
  // the robot wanders 6 m on the landing, but it is only half a storey up
  auto dsg = test::makeSharedDsg();
  TrajectoryBuilder traj(*dsg->graph);
  traj.walk({{0.0, 0.0, 0.0},
             {8.0, 0.0, 0.0},
             {11.0, 0.0, 1.65},
             {14.0, 0.0, 1.65},
             {14.0, 3.0, 1.65},
             {11.0, 3.0, 1.65},
             {8.0, 3.0, 3.3},
             {0.0, 3.0, 3.3}});

  UpdateFloorsFunctor functor(testConfig());
  spin(functor, *dsg);

  const auto& floors = dsg->graph->getLayer(DsgLayers::FLOORS);
  ASSERT_EQ(floors.numNodes(), 2u);
  ASSERT_EQ(dsg->graph->getLayer(DsgLayers::CONNECTORS).numNodes(), 1u);
  const auto& connector =
      dsg->graph->getNode("X0"_id).attributes<ConnectorNodeAttributes>();
  ASSERT_EQ(connector.landing_z.size(), 1u) << connector;
  EXPECT_NEAR(connector.landing_z[0], 1.65, 0.05);
}

TEST(UpdateFloorsFunctor, Ramp) {
  auto dsg = test::makeSharedDsg();
  TrajectoryBuilder traj(*dsg->graph);
  // a storey (2.5 m) climbed at 8 deg
  const double run = 2.5 / std::tan(8.0 * M_PI / 180.0);
  traj.walk({{0.0, 0.0, 0.0},
             {6.0, 0.0, 0.0},
             {6.0 + run, 0.0, 2.5},
             {12.0 + run, 0.0, 2.5}});

  UpdateFloorsFunctor functor(testConfig());
  spin(functor, *dsg);

  ASSERT_EQ(dsg->graph->getLayer(DsgLayers::FLOORS).numNodes(), 2u);
  ASSERT_EQ(dsg->graph->getLayer(DsgLayers::CONNECTORS).numNodes(), 1u);
  const auto& connector =
      dsg->graph->getNode("X0"_id).attributes<ConnectorNodeAttributes>();
  EXPECT_EQ(connector.connector_type, Type::RAMP);
  EXPECT_NEAR(connector.slope_deg, 8.0, 1.5);
  EXPECT_TRUE(connector.landing_z.empty());
}

TEST(UpdateFloorsFunctor, MeshDistinguishesStairsFromRamp) {
  // the same 12 deg trajectory over a stepped mesh and over a smooth slope
  for (const bool stepped : {true, false}) {
    auto dsg = test::makeSharedDsg();
    auto& graph = *dsg->graph;
    const double rise = 2.6;
    const double run = rise / std::tan(12.0 * M_PI / 180.0);
    TrajectoryBuilder traj(graph);
    traj.walk({{0.0, 0.0, 0.0},
               {6.0, 0.0, 0.0},
               {6.0 + run, 0.0, rise},
               {12.0 + run, 0.0, rise}});

    auto mesh = std::make_shared<Mesh>(false, false, false);
    const size_t num_steps = 16;
    for (size_t i = 0; i < num_steps; ++i) {
      const double x0 = 6.0 + run * i / num_steps;
      const double x1 = 6.0 + run * (i + 1) / num_steps;
      const double z0 = stepped ? rise * (i + 1) / num_steps : rise * i / num_steps;
      const double z1 = stepped ? z0 : rise * (i + 1) / num_steps;
      const size_t base = mesh->points.size();
      mesh->points.emplace_back(x0, -0.6, z0);
      mesh->points.emplace_back(x1, -0.6, z1);
      mesh->points.emplace_back(x1, 0.6, z1);
      mesh->points.emplace_back(x0, 0.6, z0);
      mesh->faces.push_back({base, base + 1, base + 2});
      mesh->faces.push_back({base, base + 2, base + 3});
    }
    graph.setMesh(mesh);

    UpdateFloorsFunctor functor(testConfig());
    spin(functor, *dsg);

    ASSERT_EQ(graph.getLayer(DsgLayers::CONNECTORS).numNodes(), 1u);
    const auto& connector = graph.getNode("X0"_id).attributes<ConnectorNodeAttributes>();
    EXPECT_TRUE(connector.source_flags & ConnectorNodeAttributes::GEOMETRY);
    if (stepped) {
      EXPECT_EQ(connector.connector_type, Type::STAIRS);
      EXPECT_NEAR(connector.step_height, rise / num_steps, 0.02);
      EXPECT_EQ(connector.num_steps, num_steps);
    } else {
      EXPECT_EQ(connector.connector_type, Type::RAMP);
    }
  }
}

TEST(UpdateFloorsFunctor, IdsStableUnderNoise) {
  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;
  addStairsScene(graph);

  UpdateFloorsFunctor functor(testConfig());
  spin(functor, *dsg);
  ASSERT_TRUE(graph.hasNode("F0"_id));
  ASSERT_TRUE(graph.hasNode("F1"_id));
  ASSERT_TRUE(graph.hasNode("X0"_id));

  const auto& agents = *graph.findLayer(2, 'a');
  for (size_t round = 0; round < 20; ++round) {
    size_t i = 0;
    for (const auto& node : agents.nodes()) {
      // deterministic jitter of up to 5 cm
      const double noise = 0.05 * std::sin(0.7 * (i++) + round);
      node.attributes().position.z() += noise;
    }
    spin(functor, *dsg, 1);
    i = 0;
    for (const auto& node : agents.nodes()) {
      node.attributes().position.z() -= 0.05 * std::sin(0.7 * (i++) + round);
    }

    EXPECT_EQ(graph.getLayer(DsgLayers::FLOORS).numNodes(), 2u) << "round " << round;
    EXPECT_TRUE(graph.hasNode("F0"_id));
    EXPECT_TRUE(graph.hasNode("F1"_id));
    EXPECT_TRUE(graph.hasNode("X0"_id));
  }
}

TEST(UpdateFloorsFunctor, BodyHeightIsRemoved) {
  auto dsg = test::makeSharedDsg();
  TrajectoryBuilder traj(*dsg->graph);
  traj.walk({{0.0, 0.0, 0.59}, {10.0, 0.0, 0.59}});

  auto config = testConfig();
  config.body_height = 0.59;
  UpdateFloorsFunctor functor(config);
  spin(functor, *dsg);

  ASSERT_TRUE(dsg->graph->hasNode("F0"_id));
  const auto& attrs = dsg->graph->getNode("F0"_id).attributes<FloorNodeAttributes>();
  EXPECT_NEAR(attrs.floor_z, 0.0, 0.05);
}

}  // namespace hydra
