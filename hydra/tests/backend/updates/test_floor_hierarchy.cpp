#include <gtest/gtest.h>
#include <hydra/backend/updates/update_buildings_functor.h>
#include <hydra/backend/updates/update_rooms_functor.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph.h>

#include <nlohmann/json.hpp>

#include "hydra_test/shared_dsg_fixture.h"

using namespace spark_dsg;

namespace hydra {
namespace {

void addTaggedPlace(SceneGraph& graph,
                    NodeId id,
                    const Eigen::Vector3d& pos,
                    std::optional<NodeId> floor,
                    std::optional<NodeId> connector = std::nullopt) {
  auto attrs = std::make_unique<PlaceNodeAttributes>(1.0, 0);
  attrs->position = pos;
  attrs->metadata.add(nlohmann::json{
      {"floor_id", floor ? nlohmann::json(*floor) : nlohmann::json(nullptr)},
      {"connector_id", connector ? nlohmann::json(*connector) : nlohmann::json(nullptr)}});
  graph.emplaceNode(DsgLayers::PLACES, id, std::move(attrs));
}

void addFloor(SceneGraph& graph, NodeId id, double z) {
  auto attrs = std::make_unique<FloorNodeAttributes>();
  attrs->position = Eigen::Vector3d(0.0, 0.0, z);
  attrs->floor_z = z;
  graph.emplaceNode(DsgLayers::FLOORS, id, std::move(attrs));
}

}  // namespace

TEST(FloorHierarchy, RoomsSplitByFloor) {
  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;
  addTaggedPlace(graph, "p0"_id, {0.0, 0.0, 1.0}, "F0"_id);
  addTaggedPlace(graph, "p1"_id, {1.0, 0.0, 1.0}, "F0"_id);
  addTaggedPlace(graph, "p2"_id, {0.0, 0.0, 4.0}, "F1"_id);
  addTaggedPlace(graph, "p3"_id, {2.0, 0.0, 2.0}, "F0"_id, "X0"_id);
  graph.insertEdge("p0"_id, "p1"_id);
  graph.insertEdge("p1"_id, "p2"_id);  // through the stairwell void
  graph.insertEdge("p1"_id, "p3"_id);  // onto the stairs

  UpdateRoomsFunctor::Config config;
  config.split_by_floor = true;
  UpdateRoomsFunctor functor(config);
  const auto split = functor.splitByFloor(graph.getLayer(DsgLayers::PLACES));

  EXPECT_TRUE(split->hasNode("p0"_id));
  EXPECT_TRUE(split->hasNode("p2"_id));
  EXPECT_FALSE(split->hasNode("p3"_id));  // connector places are not in rooms
  EXPECT_TRUE(split->hasEdge("p0"_id, "p1"_id));
  EXPECT_FALSE(split->hasEdge("p1"_id, "p2"_id));
  // the full graph keeps every edge
  EXPECT_TRUE(graph.hasEdge("p1"_id, "p2"_id));
}

TEST(FloorHierarchy, RoomsTaggedWithMajorityFloor) {
  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;
  addTaggedPlace(graph, "p0"_id, {0.0, 0.0, 1.0}, "F0"_id);
  addTaggedPlace(graph, "p1"_id, {1.0, 0.0, 1.0}, "F0"_id);
  addTaggedPlace(graph, "p2"_id, {0.0, 0.0, 4.0}, "F1"_id);
  graph.emplaceNode(DsgLayers::ROOMS, "R0"_id, std::make_unique<RoomNodeAttributes>());
  for (const auto place : {"p0"_id, "p1"_id, "p2"_id}) {
    graph.insertEdge("R0"_id, place, nullptr, true);
  }

  UpdateRoomsFunctor functor(UpdateRoomsFunctor::Config{});
  functor.tagRoomFloors(graph);
  const auto& tags = graph.getNode("R0"_id).attributes().metadata.get();
  EXPECT_EQ(tags["floor_id"].get<NodeId>(), "F0"_id);
}

TEST(FloorHierarchy, BuildingOwnsFloorsWhichOwnRooms) {
  auto dsg = test::makeSharedDsg();
  auto& graph = *dsg->graph;
  addFloor(graph, "F0"_id, 0.0);
  addFloor(graph, "F1"_id, 3.3);

  for (const auto& [room, floor] : {std::pair{"R0"_id, "F0"_id}, {"R1"_id, "F1"_id}}) {
    auto attrs = std::make_unique<RoomNodeAttributes>();
    attrs->metadata.add(nlohmann::json{{"floor_id", static_cast<NodeId>(floor)}});
    graph.emplaceNode(DsgLayers::ROOMS, room, std::move(attrs));
  }

  // stairs between the floors, touching a place of each room
  graph.emplaceNode(
      DsgLayers::CONNECTORS, "X0"_id, std::make_unique<ConnectorNodeAttributes>());
  graph.insertEdge("F0"_id, "X0"_id);
  graph.insertEdge("F1"_id, "X0"_id);
  addTaggedPlace(graph, "p0"_id, {0.0, 0.0, 1.0}, "F0"_id);
  addTaggedPlace(graph, "p1"_id, {0.0, 0.0, 4.3}, "F1"_id);
  addTaggedPlace(graph, "p2"_id, {1.0, 0.0, 2.0}, "F0"_id, "X0"_id);
  graph.insertEdge("R0"_id, "p0"_id, nullptr, true);
  graph.insertEdge("R1"_id, "p1"_id, nullptr, true);
  graph.insertEdge("X0"_id, "p2"_id, nullptr, true);
  graph.insertEdge("p0"_id, "p2"_id);
  graph.insertEdge("p2"_id, "p1"_id);

  UpdateBuildingsFunctor functor(UpdateBuildingsFunctor::Config{});
  UpdateInfo::ConstPtr info(new UpdateInfo{});
  functor.call(*graph.clone(), *dsg, info);

  ASSERT_TRUE(graph.hasNode("B0"_id));
  EXPECT_EQ(graph.getNode("F0"_id).getParent(), std::optional<NodeId>("B0"_id));
  EXPECT_EQ(graph.getNode("F1"_id).getParent(), std::optional<NodeId>("B0"_id));
  EXPECT_EQ(graph.getNode("R0"_id).getParent(), std::optional<NodeId>("F0"_id));
  EXPECT_EQ(graph.getNode("R1"_id).getParent(), std::optional<NodeId>("F1"_id));
  EXPECT_TRUE(graph.hasEdge("R0"_id, "X0"_id));
  EXPECT_TRUE(graph.hasEdge("R1"_id, "X0"_id));
  EXPECT_FALSE(graph.hasEdge("B0"_id, "R0"_id));
}

}  // namespace hydra
