#include <config_utilities/config.h>
#include <config_utilities/factory.h>
#include <gtest/gtest.h>
#include <hydra/backend/dsg_updater.h>

#include "hydra_test/shared_dsg_fixture.h"

namespace hydra {
namespace {

std::vector<std::string> call_order;
std::vector<std::string> cleanup_order;

struct OrderRecordingFunctor : public UpdateFunctor {
  struct Config {
    std::string label;
  } const config;

  explicit OrderRecordingFunctor(const Config& config) : config(config) {}

  Hooks hooks() const override {
    Hooks hooks;
    hooks.cleanup = [this](const auto&, auto&, auto*) {
      cleanup_order.push_back(config.label);
    };
    return hooks;
  }

  void call(const spark_dsg::SceneGraph&,
            SharedDsgInfo&,
            const UpdateInfo::ConstPtr&) const override {
    call_order.push_back(config.label);
  }
};

void declare_config(OrderRecordingFunctor::Config& config) {
  config::name("OrderRecordingFunctor::Config");
  config::field(config.label, "label");
}

static const auto reg = config::RegistrationWithConfig<UpdateFunctor,
                                                       OrderRecordingFunctor,
                                                       OrderRecordingFunctor::Config>(
    "OrderRecordingFunctor");

}  // namespace

TEST(DsgUpdater, FunctorsRunInConfigOrder) {
  // names deliberately sort the other way round from the config order
  DsgUpdater::Config config;
  for (const auto& label : {"zfloors", "rooms", "buildings"}) {
    config.update_functors.emplace_back(label, OrderRecordingFunctor::Config{label});
  }

  auto source = test::makeSharedDsg();
  auto target = test::makeSharedDsg();
  DsgUpdater updater(config, source->graph, target);

  call_order.clear();
  cleanup_order.clear();
  updater.callUpdateFunctions(0, std::make_shared<UpdateInfo>());

  const std::vector<std::string> expected{"zfloors", "rooms", "buildings"};
  EXPECT_EQ(call_order, expected);
  // every cleanup hook runs exactly once, after all functors
  EXPECT_EQ(cleanup_order, expected);
}

}  // namespace hydra
