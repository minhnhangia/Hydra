/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hydra/backend/updates/update_buildings_functor.h"

#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <config_utilities/validation.h>
#include <glog/logging.h>
#include <spark_dsg/node_symbol.h>
#include <spark_dsg/scene_graph.h>

#include <set>

#include "hydra/backend/updates/update_floors_functor.h"

using namespace spark_dsg;

namespace hydra {
namespace {

static const auto reg = config::RegistrationWithConfig<UpdateFunctor,
                                                       UpdateBuildingsFunctor,
                                                       UpdateBuildingsFunctor::Config>(
    "UpdateBuildingsFunctor");

}

void declare_config(UpdateBuildingsFunctor::Config&) {
  config::name("UpdateBuildingsFunctor::Config");
}

UpdateBuildingsFunctor::UpdateBuildingsFunctor(const Config&) {}

void UpdateBuildingsFunctor::call(const SceneGraph&,
                                  SharedDsgInfo& dsg,
                                  const UpdateInfo::ConstPtr&) const {
  const NodeSymbol building_id('B', 0);
  const auto floors = dsg.graph->findLayer(DsgLayers::FLOORS);
  if (floors && floors->numNodes()) {
    updateWithFloors(*dsg.graph, building_id);
    return;
  }

  const auto& rooms = dsg.graph->getLayer(DsgLayers::ROOMS);

  if (!rooms.numNodes()) {
    if (dsg.graph->hasNode(building_id)) {
      dsg.graph->removeNode(building_id);
    }

    return;
  }

  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto& node : rooms.nodes()) {
    centroid += node.attributes().position;
  }
  centroid /= rooms.numNodes();

  if (!dsg.graph->hasNode(building_id)) {
    NodeAttributes::Ptr attrs(new NodeAttributes());
    attrs->position = centroid;
    dsg.graph->emplaceNode(DsgLayers::BUILDINGS, building_id, std::move(attrs));
  } else {
    dsg.graph->getNode(building_id).attributes().position = centroid;
  }

  for (const auto& node : rooms.nodes()) {
    // add an edge while enforcing single parent
    dsg.graph->insertEdge(building_id, node.id, nullptr, true);
  }
}

void UpdateBuildingsFunctor::updateWithFloors(SceneGraph& graph, NodeId building_id) const {
  // building -> floors
  const auto& floors = graph.getLayer(DsgLayers::FLOORS);
  Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
  for (const auto& node : floors.nodes()) {
    centroid += node.attributes().position;
  }
  centroid /= floors.numNodes();

  if (!graph.hasNode(building_id)) {
    NodeAttributes::Ptr attrs(new NodeAttributes());
    attrs->position = centroid;
    graph.emplaceNode(DsgLayers::BUILDINGS, building_id, std::move(attrs));
  } else {
    graph.getNode(building_id).attributes().position = centroid;
  }

  for (const auto& node : floors.nodes()) {
    graph.insertEdge(building_id, node.id, nullptr, true);
  }

  // floors -> rooms (rooms are tagged with their floor by UpdateRoomsFunctor)
  const auto rooms = graph.findLayer(DsgLayers::ROOMS);
  if (!rooms) {
    return;
  }

  std::vector<NodeId> orphans;
  for (const auto& room : rooms->nodes()) {
    const auto floor = getFloorTag(room.attributes(), "floor_id");
    if (floor && graph.hasNode(*floor)) {
      graph.insertEdge(*floor, room.id, nullptr, true);
    } else {
      orphans.push_back(room.id);
    }
  }

  for (const auto room : orphans) {
    // without a floor, a room still belongs to the building
    graph.insertEdge(building_id, room, nullptr, true);
  }

  // rooms <-> connectors where a room's place is next to a connector's place
  const auto connectors = graph.findLayer(DsgLayers::CONNECTORS);
  if (!connectors) {
    return;
  }

  for (const auto& connector : connectors->nodes()) {
    std::set<NodeId> touching;
    for (const auto child : connector.children()) {
      for (const auto neighbor : graph.getNode(child).siblings()) {
        const auto room = graph.getNode(neighbor).getParent();
        if (room && rooms->hasNode(*room)) {
          touching.insert(*room);
        }
      }
    }

    for (const auto room : touching) {
      graph.insertEdge(room, connector.id);
    }
  }
}

}  // namespace hydra
