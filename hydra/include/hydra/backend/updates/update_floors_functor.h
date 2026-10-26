#pragma once
#include <spark_dsg/node_attributes.h>

#include <Eigen/Dense>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "hydra/backend/update_functions.h"

namespace hydra {

/**
 * @brief Segments the scene graph into floors and finds the connectors (stairs, ramps)
 * between them.
 *
 * Floors are height plateaus of walkable support, taken from the robot trajectory and
 * from the floor beneath each place. Connectors are trajectory segments with sustained
 * vertical motion between two confirmed floors; their type comes from the mesh (level
 * treads vs. a smooth slope), the semantic labels and the trajectory slope. Every spin,
 * places are tagged with their floor or connector (node metadata `floor_id` /
 * `connector_id`) and place edges joining two floors outside a connector are tagged
 * `cross_floor`, because the backend merge rewrites both on every update.
 */
struct UpdateFloorsFunctor : public UpdateFunctor {
  using Polygon = std::vector<Eigen::Vector2d>;

  struct Config {
    //! Layer of free-space places (used for support heights and connector width)
    std::string places_layer = spark_dsg::DsgLayers::PLACES;
    //! Layers whose nodes are assigned to floors, with the height of the node above
    //! the walkable surface (e.g. body height for traversability on a legged robot)
    std::map<std::string, double> assigned_layers{
        {spark_dsg::DsgLayers::PLACES, 0.0},
        {spark_dsg::DsgLayers::MESH_PLACES, 0.0},
        {spark_dsg::DsgLayers::TRAVERSABILITY, 0.0}};
    //! Height of the robot frame above the ground it stands on [m]
    double body_height = 0.0;

    //! Horizontal window for the trajectory slope fit [m]
    double slope_window_m = 1.2;
    //! A trajectory sample is flat below these slope and pitch angles [deg]
    double flat_slope_deg = 6.0;
    double flat_pitch_deg = 8.0;
    //! A trajectory sample is on a transition above these angles [deg]
    double transition_slope_deg = 8.0;
    double transition_pitch_deg = 10.0;
    //! Minimum height change for a transition between levels [m]
    double min_transition_rise_m = 0.4;

    //! Bin size and smoothing of the support-height histogram [m]
    double histogram_resolution_m = 0.05;
    double histogram_sigma_m = 0.06;
    //! Minimum height between two levels [m]
    double min_level_separation_m = 0.8;
    //! Minimum histogram weight of a level (1 = 1 m of flat trajectory)
    double min_peak_weight = 1.0;
    //! Histogram weight of one place supported by the surface below it (0 = off; GVD
    //! places over furniture make spurious peaks)
    double place_weight = 0.0;
    //! Samples within this height of a level belong to it [m]
    double level_tolerance_m = 0.3;
    //! Levels closer than this are merged [m]
    double level_merge_distance_m = 0.25;
    //! A level can become a floor with this much floor-labelled mesh [m^2]...
    double min_floor_area_m2 = 12.0;
    //! ...or this much flat trajectory outside every connector [m]
    double min_exit_path_m = 3.0;
    //! Floors are at least one storey apart; levels in between are landings [m]
    double min_storey_height_m = 2.2;
    //! A confirmed floor only gives way to a level with this many times its support
    double demote_ratio = 3.0;
    //! Spins a level must meet the floor criteria before it is confirmed
    size_t confirm_spins = 3;
    //! Spins an unconfirmed level survives without support
    size_t max_missed_spins = 50;
    //! Tolerance below a floor when assigning nodes to it [m]
    double assignment_tolerance_m = 0.2;

    //! Buffer around the trajectory hull for connector footprints [m]
    double connector_buffer_m = 0.6;
    //! Two traversals of the same floor pair closer than this are one connector [m]
    double connector_match_distance_m = 3.0;
    //! Nodes inside a connector must be at least this far from its floors [m]
    double connector_support_margin_m = 0.2;

    //! Mesh labels of stairs, given directly or by name in the active labelspace
    std::vector<uint32_t> stairs_labels;
    std::vector<std::string> stairs_label_names{"stairs", "stairs_or_slope"};
    //! Mesh labels of floor, given directly or by name in the active labelspace
    std::vector<uint32_t> floor_labels;
    std::vector<std::string> floor_label_names{"floor"};
    //! Recompute mesh evidence every N spins (0 to disable)
    size_t mesh_every_n = 10;
    //! Faces with |n_z| above these are level / walkable
    double level_normal_z = 0.995;
    double walkable_normal_z = 0.7;
    //! Minimum walkable mesh area inside a connector to use mesh evidence [m^2]
    double min_mesh_area_m2 = 1.0;
    //! Stairs: share of walkable area that is level, tread count and step height
    double min_stairs_level_ratio = 0.5;
    size_t min_tread_peaks = 3;
    double min_step_height_m = 0.10;
    double max_step_height_m = 0.24;
    //! Ramp: share of walkable area that is level and slope range [deg]
    double max_ramp_level_ratio = 0.3;
    double min_ramp_slope_deg = 3.0;
    double max_ramp_slope_deg = 15.0;
    //! Minimum sloped (walkable, not level) mesh area to judge the surface slope [m^2]
    double min_sloped_area_m2 = 0.5;
    //! Stairs if the sloped mesh surface is at least this steep [deg]. Voxels coarser
    //! than a tread smooth stairs into a slope near (but below) the stair pitch.
    double min_stairs_surface_slope_deg = 17.0;
    //! Stairs from labels or from a steep trajectory alone
    double min_stairs_label_share = 0.3;
    double min_stairs_slope_deg = 20.0;
  } const config;

  //! Trajectory sample (robot pose reduced to the ground beneath it)
  struct Sample {
    enum class Kind { FLAT, TRANSITION, UNKNOWN };
    spark_dsg::NodeId id;
    uint64_t stamp_ns;
    Eigen::Vector3d ground;  //!< position of the ground beneath the robot
    double s = 0.0;          //!< horizontal arc length
    double ds = 0.0;         //!< horizontal path length represented by the sample
    double slope = 0.0;      //!< dz/ds (NaN if the robot barely moved)
    double pitch = 0.0;      //!< body pitch [rad]
    Kind kind = Kind::UNKNOWN;
    int level = -1;  //!< index into levels_ (or -1)
  };

  //! Height plateau that may be a floor or a landing
  struct Level {
    double z = 0.0;
    double weight = 0.0;
    bool confirmed = false;
    size_t streak = 0;
    size_t missed = 0;
    uint64_t first_seen_ns = 0;
    std::optional<spark_dsg::NodeId> node_id;  //!< set once confirmed
    // statistics of the current spin
    double area_m2 = 0.0;
    double path_m = 0.0;
    double exit_path_m = 0.0;
    double mesh_area_m2 = 0.0;  //!< floor-labelled mesh at this height (throttled)
    size_t num_places = 0;
    Polygon footprint;
  };

  //! Mesh evidence about a connector
  struct MeshEvidence {
    double walkable_area = 0.0;
    double level_ratio = 0.0;
    double stairs_share = 0.0;
    size_t num_treads = 0;
    double step_height = 0.0;
    double sloped_area = 0.0;      //!< walkable faces that are not level
    double sloped_slope_deg = 0.0;  //!< their area-weighted mean slope
  };

  //! Connector between two confirmed floors
  struct Connector {
    spark_dsg::NodeId node_id = 0;
    int lower = -1;
    int upper = -1;
    std::vector<size_t> samples;  //!< transition and landing samples
    spark_dsg::ConnectorNodeAttributes attrs;
    Polygon footprint;
    std::optional<MeshEvidence> mesh;
  };

  explicit UpdateFloorsFunctor(const Config& config);

  void call(const spark_dsg::SceneGraph& unmerged,
            SharedDsgInfo& dsg,
            const UpdateInfo::ConstPtr& info) const override;

  const std::vector<Level>& levels() const { return levels_; }
  const std::vector<Connector>& connectors() const { return connectors_; }

 protected:
  std::vector<Sample> extractTrajectory(const spark_dsg::SceneGraph& graph) const;
  void classifySamples(std::vector<Sample>& samples) const;
  std::map<spark_dsg::NodeId, double> estimatePlaceSupport(
      const spark_dsg::SceneGraph& graph) const;
  void updateLevels(std::vector<Sample>& samples,
                    const std::map<spark_dsg::NodeId, double>& place_support,
                    const spark_dsg::SceneGraph& graph,
                    uint64_t timestamp_ns) const;
  std::vector<Polygon> transitionFootprints(const std::vector<Sample>& samples) const;
  void updateConnectors(const std::vector<Sample>& samples,
                        const spark_dsg::SceneGraph& graph) const;
  MeshEvidence computeMeshEvidence(const spark_dsg::Mesh& mesh,
                                   const Connector& connector) const;
  void classifyConnector(Connector& connector) const;
  void updateMeshFloorArea(const spark_dsg::Mesh& mesh) const;
  void selectFloors() const;
  void writeGraph(spark_dsg::SceneGraph& graph,
                  const std::map<spark_dsg::NodeId, double>& place_support) const;

  std::set<uint32_t> stairs_labels_;
  std::set<uint32_t> floor_labels_;
  mutable size_t num_spins_ = 0;
  mutable size_t next_floor_index_ = 0;
  mutable size_t next_connector_index_ = 0;
  mutable std::vector<Level> levels_;
  mutable std::vector<Connector> connectors_;
  mutable std::vector<Sample> samples_;
};

void declare_config(UpdateFloorsFunctor::Config& config);

/**
 * @brief Read a floor tag written by UpdateFloorsFunctor ("floor_id" or "connector_id")
 */
std::optional<spark_dsg::NodeId> getFloorTag(const spark_dsg::NodeAttributes& attrs,
                                             const std::string& key);

//! Whether UpdateFloorsFunctor tagged the edge as joining two floors outside a connector
bool isCrossFloorEdge(const spark_dsg::EdgeAttributes& attrs);

}  // namespace hydra
