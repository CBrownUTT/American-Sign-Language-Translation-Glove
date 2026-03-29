#pragma once

namespace RFModel {
  constexpr int NUM_FEATURES = 284;
  constexpr int NUM_CLASSES = 36;
  constexpr int NUM_TREES = 300;

  extern const char* FEATURE_NAMES[NUM_FEATURES];
  extern const char* CLASS_NAMES[NUM_CLASSES];

  int predict(const float *x);
  const char* class_name(int class_idx);
}
