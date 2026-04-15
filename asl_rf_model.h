#ifndef ASL_RF_MODEL_H
#define ASL_RF_MODEL_H

namespace ASLModel {

static constexpr int kFeatureCount = 284;
static constexpr int kClassCount = 36;
static constexpr int kTreeCount = 300;

extern const char* const kClassLabels[kClassCount];

int predict(const float* features);
void predict_proba(const float* features, float* out_proba);

}  // namespace ASLModel

#endif
