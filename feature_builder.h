#ifndef FEATURE_BUILDER_H
#define FEATURE_BUILDER_H

#include <stdint.h>
#include <stddef.h>

namespace ASLModel {

static constexpr int SENSOR_COUNT = 18;
static constexpr int FEATURE_COUNT = 284;

struct Frame {
    float sensor[SENSOR_COUNT];
    uint32_t timestamp_ms;
};

extern const char* const kSensorNames[SENSOR_COUNT];
extern const char* const kFeatureNames[FEATURE_COUNT];

bool build_feature_vector(const Frame* frames, int frame_count, float* out_features);

}  // namespace ASLModel

#endif
